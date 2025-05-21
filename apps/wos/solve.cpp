

#include "args.hxx"
#include "geometrycentral/surface/halfedge_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "igl/bfs_orient.h"
#include "igl/readMSH.h"
#include "igl/readOFF.h"
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <zombie/zombie.h>
#ifdef USE_FEM
#include "fem.h"
#endif

#include <chrono>

#include "build_tree.h"
#include "convert_tree.h"
#include "solve_bonsai.h"

std::string filename = "";
bool flipMeshOrientation = false;                        // input
bool domainIsOpen = false;                               // input
bool solveDoubleSided = false;                           // input
bool estimateGradients = false;                          // checkbox
bool disableGradientControlVariates = false;             // checkbox
bool disableGradientAntitheticVariates = false;          // checkbox
bool useCosineSamplingForDirectionalDerivatives = false; // checkbox
bool ignoreDirichletContribution = false;                // checkbox
bool ignoreSourceContribution = false;                   // checkbox
bool runSingleThreaded = false;                          // checkbox
bool printLogs = false;                                  // checkbox
bool estimateOnSlicePlane = false;                       // checkbox
bool useSdfForBoundary = false;
int nWalksForSamplePts = 1000;       // slider
int maxWalkLength = 10000;           // slider
int stepsBeforeApplyingTikhonov = 0; // slider
int nWalksTakenForSamplePts = 0;
int slicePlaneResolution = 3; // slider
int sdfGridResolution = 128;
float absorptionCoeff = 0.0f;          // slider
float epsilonShell = 1e-3f;            // slider
float russianRouletteThreshold = 0.0f; // slider
float splittingThreshold = 1.5f;       // slider
float boundaryDistanceMask = 0.0f;     // slider

Box box; // set up in visualize.
_tree_layout0 tree;

#ifdef USE_FEM
float femMeshRefinementMultiplier = 1.0f; // slider
#endif

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <size_t DIM>
using Vectori = Eigen::Matrix<int, DIM, 1>;
using Vector2i = Vectori<2>;
using Vector3i = Vectori<3>;

template <size_t DIM>
void loadVolumeMesh(const std::string &filename, bool flipMeshOrientation,
                    std::vector<Vector<DIM>> &meshPositions,
                    std::vector<std::vector<size_t>> &meshIndices,
                    std::vector<Vector<DIM>> &boundaryPositions,
                    std::vector<Vectori<DIM>> &boundaryIndices,
                    std::vector<bool> &isBoundaryVertex) {
    // do nothing
}

template <>
void loadVolumeMesh<2>(const std::string &filename, bool flipMeshOrientation,
                       std::vector<Vector2> &meshPositions,
                       std::vector<std::vector<size_t>> &meshIndices,
                       std::vector<Vector2> &boundaryPositions,
                       std::vector<Vector2i> &boundaryIndices,
                       std::vector<bool> &isBoundaryVertex) {
    // load mesh and geometry
    using namespace geometrycentral::surface;
    std::unique_ptr<HalfedgeMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = loadMesh(filename);
    VertexData<size_t> vIndex = mesh->getVertexIndices();
    size_t V = mesh->nVertices();

    // collect mesh positions and indices
    meshPositions.resize(V);
    meshIndices = mesh->getFaceVertexList();
    isBoundaryVertex.resize(V, false);

    for (Vertex v : mesh->vertices()) {
        size_t i = vIndex[v];
        const geometrycentral::Vector3 &p = geometry->inputVertexPositions[v];

        meshPositions[i] = Vector2(p.x, p.y);
    }

    // normalize mesh positions
    zombie::normalize<2>(meshPositions);

    for (Vertex v : mesh->vertices()) {
        size_t i = vIndex[v];
        isBoundaryVertex[i] = v.isBoundary();
    }

    // extract the boundary
    boundaryPositions.clear();
    boundaryIndices.clear();
    size_t i = 0;

    for (size_t n = 0; n < mesh->nBoundaryLoops(); n++) {
        // count boundary vertices
        size_t V = 0;
        size_t L = i;
        BoundaryLoop b = mesh->boundaryLoop(n);
        for (Vertex v : b.adjacentVertices()) {
            V++;
        }

        // collect boundary positions and indices
        for (Vertex v : b.adjacentVertices()) {
            size_t j = ((i + 1) - L) % V;

            boundaryPositions.emplace_back(meshPositions[vIndex[v]]);
            boundaryIndices.emplace_back(Vector2i(i, L + j));
            i++;
        }
    }

    if (flipMeshOrientation) {
        // flip mesh orientation
        zombie::flipOrientation<2>(boundaryIndices);
    }
}

template <>
void loadVolumeMesh<3>(const std::string &filename, bool flipMeshOrientation,
                       std::vector<Vector3> &meshPositions,
                       std::vector<std::vector<size_t>> &meshIndices,
                       std::vector<Vector3> &boundaryPositions,
                       std::vector<Vector3i> &boundaryIndices,
                       std::vector<bool> &isBoundaryVertex) {
    Eigen::MatrixXd V;   // vertex positions
    Eigen::MatrixXi Tri; // triangle indices
    Eigen::MatrixXi Tet; // tet indices
    Eigen::VectorXi TriTag;
    Eigen::VectorXi TetTag;
    bool success = filename.find("msh") != std::string::npos
                       ? igl::readMSH(filename, V, Tri, Tet, TriTag, TetTag)
                       : igl::readOFF(filename, V, Tet);
    if (!success) {
        std::cerr << "Unable to load file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<size_t>> triangleIndices;
    std::map<std::vector<size_t>, int> triangleCountMap;
    std::unordered_map<size_t, size_t> vertexIndexMap;
    meshPositions.resize(V.rows());
    meshIndices.resize(Tet.rows());
    isBoundaryVertex.resize(V.rows(), false);
    boundaryPositions.clear();
    boundaryIndices.clear();

    // collect mesh positions
    for (int i = 0; i < (int)V.rows(); i++) {
        meshPositions[i] = Vector3(V(i, 0), V(i, 1), V(i, 2));
    }

    // normalize mesh positions
    zombie::normalize<3>(meshPositions);

    // collect mesh indices; also count the number of times each triangle occurs
    // in the tet mesh
    for (int i = 0; i < (int)Tet.rows(); i++) {
        meshIndices[i] =
            std::vector<size_t>{(size_t)Tet(i, 0), (size_t)Tet(i, 1),
                                (size_t)Tet(i, 2), (size_t)Tet(i, 3)};

        triangleIndices.emplace_back(std::vector<size_t>{
            (size_t)Tet(i, 0), (size_t)Tet(i, 1), (size_t)Tet(i, 2)});
        triangleIndices.emplace_back(std::vector<size_t>{
            (size_t)Tet(i, 1), (size_t)Tet(i, 2), (size_t)Tet(i, 3)});
        triangleIndices.emplace_back(std::vector<size_t>{
            (size_t)Tet(i, 2), (size_t)Tet(i, 3), (size_t)Tet(i, 0)});
        triangleIndices.emplace_back(std::vector<size_t>{
            (size_t)Tet(i, 3), (size_t)Tet(i, 0), (size_t)Tet(i, 1)});
        std::sort(triangleIndices[4 * i + 0].begin(),
                  triangleIndices[4 * i + 0].end());
        std::sort(triangleIndices[4 * i + 1].begin(),
                  triangleIndices[4 * i + 1].end());
        std::sort(triangleIndices[4 * i + 2].begin(),
                  triangleIndices[4 * i + 2].end());
        std::sort(triangleIndices[4 * i + 3].begin(),
                  triangleIndices[4 * i + 3].end());
        triangleCountMap[triangleIndices[4 * i + 0]] =
            triangleCountMap[triangleIndices[4 * i + 0]] + 1;
        triangleCountMap[triangleIndices[4 * i + 1]] =
            triangleCountMap[triangleIndices[4 * i + 1]] + 1;
        triangleCountMap[triangleIndices[4 * i + 2]] =
            triangleCountMap[triangleIndices[4 * i + 2]] + 1;
        triangleCountMap[triangleIndices[4 * i + 3]] =
            triangleCountMap[triangleIndices[4 * i + 3]] + 1;
    }

    // extract the boundary
    for (auto it = triangleCountMap.begin(); it != triangleCountMap.end();
         ++it) {
        if (it->second == 1) {
            std::vector<size_t> index = it->first;

            for (int i = 0; i < 3; i++) {
                isBoundaryVertex[index[i]] = true;

                if (vertexIndexMap.find(index[i]) == vertexIndexMap.end()) {
                    vertexIndexMap[index[i]] = boundaryPositions.size();
                    boundaryPositions.emplace_back(meshPositions[index[i]]);
                }

                index[i] = vertexIndexMap[index[i]];
            }

            boundaryIndices.emplace_back(
                Vector3i(index[0], index[1], index[2]));
        }
    }

    // consistently orient the boundary mesh
    int B = (int)boundaryIndices.size();
    Tri = Eigen::MatrixXi::Zero(B, 3);

    for (int i = 0; i < B; i++) {
        Tri(i, 0) = boundaryIndices[i][0];
        Tri(i, 1) = boundaryIndices[i][1];
        Tri(i, 2) = boundaryIndices[i][2];
    }

    Eigen::MatrixXi TriOriented;
    Eigen::VectorXi Components;
    igl::bfs_orient(Tri, TriOriented, Components);

    for (int i = 0; i < B; i++) {
        boundaryIndices[i][0] = TriOriented(i, 0);
        boundaryIndices[i][1] = TriOriented(i, 1);
        boundaryIndices[i][2] = TriOriented(i, 2);
    }

    if (flipMeshOrientation) {
        // flip mesh orientation
        zombie::flipOrientation<3>(boundaryIndices);
    }
}

zombie::EstimationQuantity getEstimationQuantity(zombie::SampleType type,
                                                 bool estimateGradients) {
    if (estimateGradients) {
        return type != zombie::SampleType::OnAbsorbingBoundary
                   ? zombie::EstimationQuantity::SolutionAndGradient
                   : zombie::EstimationQuantity::Solution;
    }

    return zombie::EstimationQuantity::Solution;
}

template <size_t DIM>
void addMeshSampleAndEvaluationPoints(
    const std::vector<Vector<DIM>> &meshPositions,
    const std::vector<bool> &isBoundaryVertex,
    const zombie::GeometricQueries<DIM> &geometricQueries,
    int nWalksForSamplePts, bool estimateGradients, std::vector<int> &nWalks,
    std::vector<zombie::SamplePoint<float, DIM>> &samplePoints) {
    int nV = (int)meshPositions.size();
    samplePoints.clear();
    nWalks.clear();
    samplePoints.reserve(nV);
    nWalks.resize(nV, nWalksForSamplePts);

    for (int i = 0; i < nV; i++) {
        const Vector<DIM> &pt = meshPositions[i];
        zombie::SampleType type = isBoundaryVertex[i]
                                      ? zombie::SampleType::OnAbsorbingBoundary
                                      : zombie::SampleType::InDomain;

        float distToAbsorbingBoundary =
            geometricQueries.computeDistToAbsorbingBoundary(pt, false);
        samplePoints.emplace_back(zombie::SamplePoint<float, DIM>(
            pt, Vector<DIM>::Zero(), type,
            getEstimationQuantity(type, estimateGradients), 1.0f,
            distToAbsorbingBoundary, 0.0f));
    }
}

template <size_t DIM>
void buildSlicePlane(int extent, float scale, const Vector<DIM> &shift,
                     std::vector<Vector<DIM>> &positions,
                     std::vector<std::vector<size_t>> &indices) {
    positions.clear();
    indices.clear();
    int nPositions = (extent + 1) * (extent + 1);
    int nIndices = extent * extent;
    positions.reserve(nPositions);
    indices.reserve(nIndices);

    for (size_t h = 0; h < extent + 1; h++) {
        for (size_t w = 0; w < extent + 1; w++) {
            size_t index = positions.size();
            positions.emplace_back(Vector<DIM>::Zero());
            positions[index](DIM == 2 ? 0 : 1) = w;
            positions[index](DIM == 2 ? 1 : 2) = h;

            if (h != extent && w != extent) {
                indices.emplace_back(std::vector<size_t>{
                    index, index + 1, index + extent + 2, index + extent + 1});
            }
        }
    }

    // center and scale the grid
    Vector<DIM> cm = Vector<DIM>::Zero();
    float radius = 0.0f;

    for (int i = 0; i < nPositions; i++) {
        cm += positions[i];
    }

    cm /= nPositions;
    for (int i = 0; i < nPositions; i++) {
        positions[i] -= cm;
        radius = std::max(radius, positions[i].norm());
    }

    for (int i = 0; i < nPositions; i++) {
        positions[i] /= radius;
        positions[i] *= scale;
        positions[i] += shift;
    }
}

template <size_t DIM>
void transformSlicePlane(const glm::mat4 &objectTransform,
                         std::vector<Vector<DIM>> &positions) {
    glm::vec3 normal{objectTransform[0][0], objectTransform[0][1],
                     objectTransform[0][2]};
    normal = glm::normalize(normal);

    for (int i = 0; i < (int)positions.size(); i++) {
        glm::vec4 coord(positions[i](0), positions[i](1), positions[i](2),
                        1.0f);
        coord = objectTransform * coord;

        positions[i](0) = coord[0] + 1e-4f * normal[0];
        positions[i](1) = coord[1] + 1e-4f * normal[1];
        positions[i](2) = coord[2] + 1e-4f * normal[2];
    }
}

template <size_t DIM>
void addSlicePlane(const Vector<DIM> &boundingBoxMin,
                   const Vector<DIM> &boundingBoxMax,
                   const glm::mat4 &objectTransform, float resolution,
                   std::vector<Vector<DIM>> &positions,
                   std::vector<std::vector<size_t>> &indices) {
    // build the slice plane
    positions.clear();
    indices.clear();
    Vector<DIM> center = 0.5f * (boundingBoxMin + boundingBoxMax);
    Vector<DIM> extent = boundingBoxMax - boundingBoxMin;
    if (DIM == 3) {
        center(0) = 0.0f;
        extent(0) = 0.0f;
    }

    buildSlicePlane<DIM>(std::pow(2, 5 + resolution), 0.475f * extent.norm(),
                         center, positions, indices);

    // transform the slice plane
    if (DIM == 3) {
        transformSlicePlane<DIM>(objectTransform, positions);
    }

    // plot
    if (DIM == 2) {
        polyscope::registerSurfaceMesh2D("Slice Plane", positions, indices);

    } else if (DIM == 3) {
        polyscope::registerSurfaceMesh("Slice Plane", positions, indices);
    }
}

template <size_t DIM>
void addSlicePlaneSampleAndEvaluationPoints(
    const std::vector<Vector<DIM>> &slicePlanePositions,
    const std::vector<std::vector<size_t>> &slicePlaneIndices,
    const zombie::GeometricQueries<DIM> &geometricQueries,
    int nWalksForSamplePts, bool estimateGradients, std::vector<int> &nWalks,
    std::vector<zombie::SamplePoint<float, DIM>> &samplePoints) {
    int nF = (int)slicePlaneIndices.size();
    samplePoints.clear();
    nWalks.clear();
    samplePoints.reserve(nF);
    nWalks.resize(nF, nWalksForSamplePts);

    for (int i = 0; i < nF; i++) {
        int V = (int)slicePlaneIndices[i].size();
        Vector<DIM> center = Vector<DIM>::Zero();

        for (int j = 0; j < V; j++) {
            size_t index = slicePlaneIndices[i][j];
            center += slicePlanePositions[index];
        }

        center /= V;
        float distToAbsorbingBoundary =
            geometricQueries.computeDistToAbsorbingBoundary(center, false);
        samplePoints.emplace_back(zombie::SamplePoint<float, DIM>(
            center, Vector<DIM>::Zero(), zombie::SampleType::InDomain,
            getEstimationQuantity(zombie::SampleType::InDomain,
                                  estimateGradients),
            1.0f, distToAbsorbingBoundary, 0.0f));
    }
}

template <size_t DIM>
void plotSamplePoints(
    const std::vector<zombie::SamplePoint<float, DIM>> &samplePoints) {
    std::vector<Vector<DIM>> points;
    std::vector<float> boundaryDistances;
    for (int i = 0; i < (int)samplePoints.size(); i++) {
        points.emplace_back(samplePoints[i].pt);
        boundaryDistances.emplace_back(samplePoints[i].distToAbsorbingBoundary);
    }

    if (DIM == 2) {
        polyscope::PointCloud *pointCloud =
            polyscope::registerPointCloud2D("Sample Points", points);
        pointCloud->addScalarQuantity("Boundary Distance", boundaryDistances);
        pointCloud->setEnabled(false);

    } else if (DIM == 3) {
        polyscope::PointCloud *pointCloud =
            polyscope::registerPointCloud("Sample Points", points);
        pointCloud->addScalarQuantity("Boundary Distance", boundaryDistances);
        pointCloud->setEnabled(false);
    }
}

template <size_t DIM>
void plotWalkOnSpheresEstimates(
    const std::vector<zombie::SamplePoint<float, DIM>> &samplePoints,
    bool estimateOnSlicePlane, std::vector<float> &solution) {
    int nPoints = (int)samplePoints.size();
    solution.clear();
    solution.resize(nPoints, 0.0f);
    std::vector<float> solutionVariance(nPoints, 0.0f);
    std::vector<std::vector<float>> gradient(nPoints,
                                             std::vector<float>(DIM, 0.0f));
    std::vector<std::vector<float>> gradientVariance(
        nPoints, std::vector<float>(DIM, 0.0f));
    std::vector<float> walkLength(nPoints, 0.0f);

    float solutionAvg = 0.0f;
    float solutionMin = INFINITY;
    float solutionMax = -INFINITY;
    float solutionVarianceAvg = 0.0f;
    float solutionVarianceMin = INFINITY;
    float solutionVarianceMax = -INFINITY;
    float nSolutionCountAvg = 0.0f;
    float gradientAvg = 0.0f;
    float gradientMin = INFINITY;
    float gradientMax = -INFINITY;
    float gradientVarianceAvg = 0.0f;
    float gradientVarianceMin = INFINITY;
    float gradientVarianceMax = -INFINITY;
    float nGradientCountAvg = 0.0f;
    float walkLengthAvg = 0.0;
    float walkLengthMin = INFINITY;
    float walkLengthMax = -INFINITY;
    float nSplitsAvg = 0.0f;

    for (int i = 0; i < nPoints; i++) {
        if (samplePoints[i].statistics.getSolutionEstimateCount() == 0) {
            solution[i] = 0.0f;
            continue;
        }

        const zombie::SampleStatistics<float, DIM> &statistics =
            samplePoints[i].statistics;
        int nSolutionCount = std::max(1, statistics.getSolutionEstimateCount());
        solution[i] = statistics.getEstimatedSolution();
        solutionVariance[i] = statistics.getEstimatedSolutionVariance();

        solutionVariance[i] /= nSolutionCount;
        solutionAvg += solution[i];
        solutionMin = std::min(solutionMin, solution[i]);
        solutionMax = std::max(solutionMax, solution[i]);
        solutionVarianceAvg += solutionVariance[i];
        solutionVarianceMin =
            std::min(solutionVarianceMin, solutionVariance[i]);
        solutionVarianceMax =
            std::max(solutionVarianceMax, solutionVariance[i]);
        nSolutionCountAvg += nSolutionCount;

        float gradientNorm = 0.0f;
        float gradientVarianceNorm = 0.0f;
        int nGradientCount = std::max(1, statistics.getGradientEstimateCount());

        for (int j = 0; j < DIM; j++) {
            gradient[i][j] = statistics.getEstimatedGradient()[j];
            gradientVariance[i][j] =
                statistics.getEstimatedGradientVariance()[j];

            gradientNorm += std::pow(gradient[i][j], 2);
            gradientVarianceNorm += std::pow(gradientVariance[i][j], 2);
        }

        gradientNorm = std::sqrt(gradientNorm);
        gradientAvg += gradientNorm;
        gradientMin = std::min(gradientMin, gradientNorm);
        gradientMax = std::max(gradientMax, gradientNorm);
        gradientVarianceNorm = std::sqrt(gradientVarianceNorm) / nGradientCount;
        gradientVarianceAvg += gradientVarianceNorm;
        gradientVarianceMin =
            std::min(gradientVarianceMin, gradientVarianceNorm);
        gradientVarianceMax =
            std::max(gradientVarianceMax, gradientVarianceNorm);
        nGradientCountAvg += nGradientCount;

        float meanWalkLength = statistics.getMeanWalkLength();
        walkLength[i] = meanWalkLength;
        walkLengthAvg += meanWalkLength;
        walkLengthMin = std::min(walkLengthMin, meanWalkLength);
        walkLengthMax = std::max(walkLengthMax, meanWalkLength);

        nSplitsAvg += statistics.getMeanSplits();
    }

    solutionAvg /= nPoints;
    solutionVarianceAvg /= nPoints;
    nSolutionCountAvg /= nPoints;
    gradientAvg /= nPoints;
    gradientVarianceAvg /= nPoints;
    nGradientCountAvg /= nPoints;
    walkLengthAvg /= nPoints;
    nSplitsAvg /= nPoints;

    std::cout << "Walk on spheres (zombie)" << std::endl;
    std::cout << "  solution avg: " << solutionAvg << " min: " << solutionMin
              << " max: " << solutionMax << std::endl;
    std::cout << "  solution variance avg: " << solutionVarianceAvg
              << " min: " << solutionVarianceMin
              << " max: " << solutionVarianceMax << std::endl;
    std::cout << "  gradient avg: " << gradientAvg << " min: " << gradientMin
              << " max: " << gradientMax << std::endl;
    std::cout << "  gradient variance avg: " << gradientVarianceAvg
              << " min: " << gradientVarianceMin
              << " max: " << gradientVarianceMax << std::endl;
    std::cout << "  mean sample count for solution: " << nSolutionCountAvg
              << " gradient: " << nGradientCountAvg << std::endl;
    std::cout << "  mean walk length: " << walkLengthAvg
              << " min: " << walkLengthMin << " max: " << walkLengthMax
              << std::endl;
    std::cout << "  mean splits performed: " << nSplitsAvg << std::endl;

    if (estimateOnSlicePlane) {
        polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Slice Plane");
        mesh->addFaceScalarQuantity("Solution Estimate", solution);
        mesh->getQuantity("Solution Estimate")->setEnabled(true);
        if (DIM == 2) {
            mesh->addFaceVectorQuantity2D("Gradient Estimate", gradient);

        } else if (DIM == 3) {
            mesh->addFaceVectorQuantity("Gradient Estimate", gradient);
        }

    } else {
        if (DIM == 2) {
            polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Mesh");
            mesh->addVertexScalarQuantity("Solution Estimate", solution);
            mesh->addVertexScalarQuantity("Solution Variance",
                                          solutionVariance);
            mesh->addVertexScalarQuantity("Mean Walk Length", walkLength);
            mesh->addVertexVectorQuantity2D("Gradient Estimate", gradient);
            mesh->getQuantity("Solution Estimate")->setEnabled(true);

        } else if (DIM == 3) {
            polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
            mesh->addVertexScalarQuantity("Solution Estimate", solution);
            mesh->addVertexScalarQuantity("Solution Variance",
                                          solutionVariance);
            mesh->addVertexScalarQuantity("Mean Walk Length", walkLength);
            mesh->addVertexVectorQuantity("Gradient Estimate", gradient);
            mesh->getQuantity("Solution Estimate")->setEnabled(true);
        }
    }
}

template <size_t DIM>
void maskWalkOnSpheresEstimates(
    float boundaryDistanceMask,
    std::vector<zombie::SamplePoint<float, DIM>> &samplePoints) {
    if (boundaryDistanceMask > 0.0f) {
        auto run = [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                if (samplePoints[i].distToAbsorbingBoundary <=
                    boundaryDistanceMask) {
                    samplePoints[i].reset();
                }
            }
        };

        int nPoints = (int)samplePoints.size();
        tbb::blocked_range<int> range(0, nPoints);
        tbb::parallel_for(range, run);
    }
}

void plotWalkOnSpheresEstimatesBonsai(const Statistics *bonsai_solution,
                                      const uint64_t nPoints,
                                      bool estimateOnSlicePlane,
                                      std::vector<float> &solution) {
    solution.clear();
    solution.resize(nPoints, 0.0f);
    std::vector<float> solutionVariance(nPoints, 0.0f);
    std::vector<std::vector<float>> gradient(nPoints,
                                             std::vector<float>(3, 0.0f));
    std::vector<std::vector<float>> gradientVariance(
        nPoints, std::vector<float>(3, 0.0f));
    std::vector<float> walkLength(nPoints, 0.0f);

    float solutionAvg = 0.0f;
    float solutionMin = INFINITY;
    float solutionMax = -INFINITY;
    float solutionVarianceAvg = 0.0f;
    float solutionVarianceMin = INFINITY;
    float solutionVarianceMax = -INFINITY;
    float nSolutionCountAvg = 0.0f;
    float gradientAvg = 0.0f;
    float gradientMin = INFINITY;
    float gradientMax = -INFINITY;
    float gradientVarianceAvg = 0.0f;
    float gradientVarianceMin = INFINITY;
    float gradientVarianceMax = -INFINITY;
    float nGradientCountAvg = 0.0f;
    float walkLengthAvg = 0.0;
    float walkLengthMin = INFINITY;
    float walkLengthMax = -INFINITY;
    float nSplitsAvg = 0.0f;

    for (int i = 0; i < nPoints; i++) {
        if (bonsai_solution[i].nSolEstimates == 0) {
            solution[i] = 0.0f;
            continue;
        }

        int nSolutionCount =
            std::max((uint32_t)1, bonsai_solution[i].nSolEstimates);
        solution[i] = bonsai_solution[i].solMean;
        solutionVariance[i] =
            bonsai_solution[i].solMean2 /
            std::max((uint32_t)1, bonsai_solution[i].nSolEstimates - 1);

        solutionVariance[i] /= nSolutionCount;
        solutionAvg += solution[i];
        solutionMin = std::min(solutionMin, solution[i]);
        solutionMax = std::max(solutionMax, solution[i]);
        solutionVarianceAvg += solutionVariance[i];
        solutionVarianceMin =
            std::min(solutionVarianceMin, solutionVariance[i]);
        solutionVarianceMax =
            std::max(solutionVarianceMax, solutionVariance[i]);
        nSolutionCountAvg += nSolutionCount;

        float gradientNorm = 0.0f;
        float gradientVarianceNorm = 0.0f;
        // int nGradientCount = std::max(1,
        // /*statistics.getGradientEstimateCount()*/0);

        // for (int j = 0; j < DIM; j++) {
        //     gradient[i][j] = statistics.getEstimatedGradient()[j];
        //     gradientVariance[i][j] =
        //     statistics.getEstimatedGradientVariance()[j];

        //     gradientNorm += std::pow(gradient[i][j], 2);
        //     gradientVarianceNorm += std::pow(gradientVariance[i][j], 2);
        // }

        // gradientNorm = std::sqrt(gradientNorm);
        // gradientAvg += gradientNorm;
        // gradientMin = std::min(gradientMin, gradientNorm);
        // gradientMax = std::max(gradientMax, gradientNorm);
        // gradientVarianceNorm =
        // std::sqrt(gradientVarianceNorm)/nGradientCount; gradientVarianceAvg
        // += gradientVarianceNorm; gradientVarianceMin =
        // std::min(gradientVarianceMin, gradientVarianceNorm);
        // gradientVarianceMax = std::max(gradientVarianceMax,
        // gradientVarianceNorm); nGradientCountAvg += nGradientCount;

        float meanWalkLength =
            bonsai_solution[i].totalWalkLength /
            std::max((uint32_t)1, bonsai_solution[i].nSolEstimates);
        walkLength[i] = meanWalkLength;
        walkLengthAvg += meanWalkLength;
        walkLengthMin = std::min(walkLengthMin, meanWalkLength);
        walkLengthMax = std::max(walkLengthMax, meanWalkLength);

        nSplitsAvg += bonsai_solution[i].totalSplits /
                      std::max((uint32_t)1, bonsai_solution[i].nSolEstimates);
    }

    solutionAvg /= nPoints;
    solutionVarianceAvg /= nPoints;
    nSolutionCountAvg /= nPoints;
    gradientAvg /= nPoints;
    gradientVarianceAvg /= nPoints;
    nGradientCountAvg /= nPoints;
    walkLengthAvg /= nPoints;
    nSplitsAvg /= nPoints;

    std::cout << "Walk on spheres (bonsai)" << std::endl;
    std::cout << "  solution avg: " << solutionAvg << " min: " << solutionMin
              << " max: " << solutionMax << std::endl;
    std::cout << "  solution variance avg: " << solutionVarianceAvg
              << " min: " << solutionVarianceMin
              << " max: " << solutionVarianceMax << std::endl;
    std::cout << "  gradient avg: " << gradientAvg << " min: " << gradientMin
              << " max: " << gradientMax << std::endl;
    std::cout << "  gradient variance avg: " << gradientVarianceAvg
              << " min: " << gradientVarianceMin
              << " max: " << gradientVarianceMax << std::endl;
    std::cout << "  mean sample count for solution: " << nSolutionCountAvg
              << " gradient: " << nGradientCountAvg << std::endl;
    std::cout << "  mean walk length: " << walkLengthAvg
              << " min: " << walkLengthMin << " max: " << walkLengthMax
              << std::endl;
    std::cout << "  mean splits performed: " << nSplitsAvg << std::endl;

    if (estimateOnSlicePlane) {
        polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Slice Plane");
        mesh->addFaceScalarQuantity("Solution Estimate", solution);
        mesh->getQuantity("Solution Estimate")->setEnabled(true);
        mesh->addFaceVectorQuantity("Gradient Estimate", gradient);
    } else {
        polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
        mesh->addVertexScalarQuantity("Solution Estimate", solution);
        mesh->addVertexScalarQuantity("Solution Variance", solutionVariance);
        mesh->addVertexScalarQuantity("Mean Walk Length", walkLength);
        mesh->addVertexVectorQuantity("Gradient Estimate", gradient);
        mesh->getQuantity("Solution Estimate")->setEnabled(true);
    }
}

template <size_t DIM>
void maskWalkOnSpheresEstimatesBonsai(
    float boundaryDistanceMask,
    const std::vector<zombie::SamplePoint<float, DIM>> &samplePoints,
    Statistics *bonsai_solution) {
    if (boundaryDistanceMask > 0.0f) {
        auto run = [&](const tbb::blocked_range<int> &range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                if (samplePoints[i].distToAbsorbingBoundary <=
                    boundaryDistanceMask) {
                    bonsai_solution[i].solMean = 0.0;
                    bonsai_solution[i].solMean2 = 0.0;
                    bonsai_solution[i].totalFirstSourceContribution = 0.0;
                    bonsai_solution[i].nSolEstimates = 0;
                    bonsai_solution[i].totalWalkLength = 0;
                    bonsai_solution[i].totalSplits = 0;
                    bonsai_solution[i].firstSphereRadius = 0.0;
                }
            }
        };

        int nPoints = (int)samplePoints.size();
        tbb::blocked_range<int> range(0, nPoints);
        tbb::parallel_for(range, run);
    }
}

#ifdef USE_FEM
template <size_t DIM>
void plotFemResults(const std::vector<Vector<DIM>> &meshPositionsRefined,
                    const std::vector<std::vector<size_t>> &meshIndicesRefined,
                    const std::vector<float> &solutionRefined,
                    const std::vector<std::vector<float>> &gradientRefined,
                    const std::vector<float> &solution,
                    const std::vector<std::vector<float>> &gradient) {
    int nPoints = (int)solution.size();
    float solutionAvg = 0.0f;
    float solutionMin = INFINITY;
    float solutionMax = -INFINITY;
    float gradientAvg = 0.0f;
    float gradientMin = INFINITY;
    float gradientMax = -INFINITY;

    for (int i = 0; i < nPoints; i++) {
        solutionAvg += solution[i];
        solutionMin = std::min(solutionMin, solution[i]);
        solutionMax = std::max(solutionMax, solution[i]);

        float gradientNorm = 0.0f;
        for (int j = 0; j < DIM; j++) {
            gradientNorm += std::pow(gradient[i][j], 2);
        }

        gradientNorm = std::sqrt(gradientNorm);
        gradientAvg += gradientNorm;
        gradientMin = std::min(gradientMin, gradientNorm);
        gradientMax = std::max(gradientMax, gradientNorm);
    }

    solutionAvg /= nPoints;
    gradientAvg /= nPoints;

    std::cout << "FEM" << std::endl;
    std::cout << "  solution avg: " << solutionAvg << " min: " << solutionMin
              << " max: " << solutionMax << std::endl;
    std::cout << "  gradient avg: " << gradientAvg << " min: " << gradientMin
              << " max: " << gradientMax << std::endl;

    if (DIM == 2) {
        polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Mesh");
        mesh->addVertexScalarQuantity("Solution", solution);
        mesh->addVertexVectorQuantity2D("Gradient", gradient);
        mesh->getQuantity("Solution")->setEnabled(true);

        polyscope::registerSurfaceMesh2D("FEM Mesh", meshPositionsRefined,
                                         meshIndicesRefined);
        polyscope::SurfaceMesh *femMesh = polyscope::getSurfaceMesh("FEM Mesh");
        femMesh->addVertexScalarQuantity("Solution", solutionRefined);
        femMesh->addVertexVectorQuantity2D("Gradient", gradientRefined);
        femMesh->setEnabled(false);

    } else if (DIM == 3) {
        polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
        mesh->addVertexScalarQuantity("Solution", solution);
        mesh->addVertexVectorQuantity("Gradient", gradient);
        mesh->getQuantity("Solution")->setEnabled(true);

        polyscope::registerTetMesh("FEM Mesh", meshPositionsRefined,
                                   meshIndicesRefined);
        polyscope::VolumeMesh *femMesh = polyscope::getVolumeMesh("FEM Mesh");
        femMesh->addVertexScalarQuantity("Solution", solutionRefined);
        femMesh->addVertexVectorQuantity("Gradient", gradientRefined);
        femMesh->setEnabled(false);
    }
}

template <size_t DIM>
void plotAbsoluteDifference(int nPoints, const std::vector<float> &femSolution,
                            const std::vector<float> &mcSolution) {
    std::vector<float> absoluteDifference(nPoints);
    for (int i = 0; i < nPoints; i++) {
        if (femSolution.size() > 0 && mcSolution.size() > 0) {
            absoluteDifference[i] = std::abs(mcSolution[i] - femSolution[i]);

        } else if (femSolution.size() > 0) {
            absoluteDifference[i] = std::abs(femSolution[i]);

        } else if (mcSolution.size() > 0) {
            absoluteDifference[i] = std::abs(mcSolution[i]);
        }
    }

    if (DIM == 2) {
        polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Mesh");
        mesh->addVertexScalarQuantity("Absolute Difference",
                                      absoluteDifference);

    } else if (DIM == 3) {
        polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
        mesh->addVertexScalarQuantity("Absolute Difference",
                                      absoluteDifference);
    }
}
#endif

template <size_t DIM>
void clearResults(std::vector<zombie::SamplePoint<float, DIM>> &samplePoints) {
    if (!domainIsOpen) {
        if (DIM == 2) {
            polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Mesh");
            mesh->removeQuantity("Solution Estimate");
            mesh->removeQuantity("Solution Variance");
            mesh->removeQuantity("Gradient Estimate");
            mesh->removeQuantity("Mean Walk Length");

        } else if (DIM == 3) {
            polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
            mesh->removeQuantity("Solution Estimate");
            mesh->removeQuantity("Solution Variance");
            mesh->removeQuantity("Gradient Estimate");
            mesh->removeQuantity("Mean Walk Length");
        }
    }

#ifdef USE_FEM
    polyscope::removeStructure("FEM Mesh", false);
    if (!domainIsOpen) {
        if (DIM == 2) {
            polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Mesh");
            mesh->removeQuantity("Solution");
            mesh->removeQuantity("Gradient");
            mesh->removeQuantity("Absolute Difference");

        } else if (DIM == 3) {
            polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
            mesh->removeQuantity("Solution");
            mesh->removeQuantity("Gradient");
            mesh->removeQuantity("Absolute Difference");
        }
    }
#endif

    if (polyscope::state::slicePlanes.size() > 0 &&
        polyscope::hasStructure("Surface Mesh", "Slice Plane")) {
        polyscope::SurfaceMesh *mesh = polyscope::getSurfaceMesh("Slice Plane");
        mesh->removeQuantity("Solution Estimate");
        mesh->removeQuantity("Gradient Estimate");
    }

    for (int i = 0; i < (int)samplePoints.size(); i++) {
        samplePoints[i].reset();
    }

    nWalksTakenForSamplePts = 0;
}

template <size_t DIM>
void guiCallback(const std::vector<Vector<DIM>> &meshPositions,
                 const std::vector<bool> &isBoundaryVertex,
                 const zombie::GeometricQueries<DIM> &geometricQueries,
                 const zombie::WalkOnSpheres<float, DIM> &walkOnSpheres,
#ifdef USE_FEM
                 zombie::FemSolver<DIM> &femSolver,
#endif
                 zombie::PDE<float, DIM> &pde, std::vector<int> &nWalks,
                 std::vector<zombie::SamplePoint<float, DIM>> &samplePoints,
                 std::vector<float> &walkOnSpheresSolution
#ifdef USE_FEM
                 ,
                 std::vector<float> &femSolution
#endif
) {
    // make ui elements 150 pixels wide, instead of full width
    ImGui::PushItemWidth(150);

    ImGui::TextUnformatted("Problem Input");
    ImGui::Indent();

    if (ImGui::Checkbox("Ignore Source Contribution",
                        &ignoreSourceContribution)) {
        clearResults<DIM>(samplePoints);
    }

    if (ImGui::Checkbox("Ignore Dirichlet Contribution",
                        &ignoreDirichletContribution)) {
        clearResults<DIM>(samplePoints);
    }

    if (ImGui::SliderFloat("Absorption Coefficient", &absorptionCoeff, 0.0f,
                           100.0f, "%.1f", 2.5f)) {
        pde.absorptionCoeff = absorptionCoeff;
        clearResults<DIM>(samplePoints);
    }

    ImGui::Unindent();

    ImGui::TextUnformatted("Common MC Settings");
    ImGui::Indent();

    ImGui::SliderFloat("Epsilon Shell", &epsilonShell, 1e-4f, 1e-2f, "%.4f",
                       2.5f);
    ImGui::SliderFloat("Russian Roulette Threshold", &russianRouletteThreshold,
                       0.0f, 0.99f, "%.2f");
    ImGui::SliderFloat("Splitting Threshold", &splittingThreshold, 1.01f, 10.0f,
                       "%.2f");
    ImGui::SliderInt("Max Walk Length", &maxWalkLength, 0, 10000);
    ImGui::SliderInt("Steps Before Applying Tikhonov",
                     &stepsBeforeApplyingTikhonov, 0, maxWalkLength);
    if (ImGui::SliderFloat("Boundary Distance Mask", &boundaryDistanceMask,
                           0.0f, 0.1f, "%.4f", 2.5f)) {
        // mask out values close to the boundary
        maskWalkOnSpheresEstimates<DIM>(boundaryDistanceMask, samplePoints);

        // plot sample results
        plotWalkOnSpheresEstimates<DIM>(samplePoints, estimateOnSlicePlane,
                                        walkOnSpheresSolution);
#ifdef USE_FEM
        if (!estimateOnSlicePlane) {
            plotAbsoluteDifference<DIM>(samplePoints.size(),
                                        walkOnSpheresSolution, femSolution);
        }
#endif
    }
    if (polyscope::state::slicePlanes.size() > 0 &&
        polyscope::hasStructure("Surface Mesh", "Slice Plane")) {
        if (ImGui::SliderInt("Slice Plane Resolution", &slicePlaneResolution, 0,
                             5)) {
            std::vector<Vector<DIM>> slicePlanePositions;
            std::vector<std::vector<size_t>> slicePlaneIndices;
            glm::mat4 currentTransform =
                polyscope::state::slicePlanes[0]->getTransform();
            addSlicePlane<DIM>(geometricQueries.domainMin,
                               geometricQueries.domainMax, currentTransform,
                               slicePlaneResolution, slicePlanePositions,
                               slicePlaneIndices);
            addSlicePlaneSampleAndEvaluationPoints<DIM>(
                slicePlanePositions, slicePlaneIndices, geometricQueries,
                nWalksForSamplePts, estimateGradients, nWalks, samplePoints);
            plotSamplePoints<DIM>(samplePoints);

            nWalksTakenForSamplePts = 0;
        }
    }
    if (!domainIsOpen) {
        if (ImGui::Checkbox("Estimate on Slice Plane", &estimateOnSlicePlane)) {
            if (estimateOnSlicePlane) {
                // initialize sample points as slice plane positions
                if (polyscope::state::slicePlanes.size() == 0) {
                    polyscope::SlicePlane *psPlane =
                        polyscope::addSceneSlicePlane();
                    psPlane->setDrawPlane(false);
                    psPlane->setDrawWidget(false);
                    psPlane->setActive(false);
                    psPlane->setPose(glm::vec3{0., 0., 0.},
                                     glm::vec3{0., 0, 1.});
                }

                std::vector<Vector<DIM>> slicePlanePositions;
                std::vector<std::vector<size_t>> slicePlaneIndices;
                glm::mat4 currentTransform =
                    polyscope::state::slicePlanes[0]->getTransform();
                addSlicePlane<DIM>(geometricQueries.domainMin,
                                   geometricQueries.domainMax, currentTransform,
                                   slicePlaneResolution, slicePlanePositions,
                                   slicePlaneIndices);
                addSlicePlaneSampleAndEvaluationPoints<DIM>(
                    slicePlanePositions, slicePlaneIndices, geometricQueries,
                    nWalksForSamplePts, estimateGradients, nWalks,
                    samplePoints);

                if (DIM == 2) {
                    polyscope::SurfaceMesh *mesh =
                        polyscope::getSurfaceMesh("Mesh");
                    mesh->removeQuantity("Solution Estimate");
                    mesh->removeQuantity("Solution Variance");
                    mesh->removeQuantity("Gradient Estimate");
                    mesh->removeQuantity("Mean Walk Length");
#ifdef USE_FEM
                    mesh->removeQuantity("Absolute Difference");
#endif

                } else if (DIM == 3) {
                    polyscope::VolumeMesh *mesh =
                        polyscope::getVolumeMesh("Mesh");
                    mesh->removeQuantity("Solution Estimate");
                    mesh->removeQuantity("Solution Variance");
                    mesh->removeQuantity("Gradient Estimate");
                    mesh->removeQuantity("Mean Walk Length");
#ifdef USE_FEM
                    mesh->removeQuantity("Absolute Difference");
#endif
                }

            } else {
                // initialize sample points as mesh positions
                addMeshSampleAndEvaluationPoints<DIM>(
                    meshPositions, isBoundaryVertex, geometricQueries,
                    nWalksForSamplePts, estimateGradients, nWalks,
                    samplePoints);

                polyscope::removeStructure("Slice Plane", false);
                polyscope::removeLastSceneSlicePlane();
            }

            plotSamplePoints<DIM>(samplePoints);

            nWalksTakenForSamplePts = 0;
        }
    }
    ImGui::Checkbox("Disable Gradient Control Variates",
                    &disableGradientControlVariates);
    ImGui::Checkbox("Disable Gradient Antithetic Variates",
                    &disableGradientAntitheticVariates);
    ImGui::Checkbox("Use Cosine Sampling For Directional Derivatives",
                    &useCosineSamplingForDirectionalDerivatives);
    ImGui::Checkbox("Run Single Threaded", &runSingleThreaded);
    ImGui::Checkbox("Print Logs", &printLogs);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (!estimateOnSlicePlane || (estimateOnSlicePlane && solveDoubleSided)) {
        ImGui::TextUnformatted("Point Estimation");
        ImGui::Indent();

        // if (ImGui::SliderInt("Walks Per Sample Point", &nWalksForSamplePts,
        // 1, 10000)) {
        //     std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // }
        nWalksForSamplePts = 1;
        std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // if (ImGui::SliderInt("Walks Per Sample Point", &nWalksForSamplePts,
        // 1, 10000)) {
        //     std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // }
        if (ImGui::Checkbox("Estimate Gradients", &estimateGradients)) {
            for (int i = 0; i < (int)samplePoints.size(); i++) {
                samplePoints[i].estimationQuantity = getEstimationQuantity(
                    samplePoints[i].type, estimateGradients);
            }

            clearResults<DIM>(samplePoints);
        }

        // solve with walk on spheres
        // if (ImGui::Button("Solve with Walk On Spheres")) {
        // estimate pointwise
        ProgressBar pb(samplePoints.size());
        std::function<void(int, int)> reportProgress =
            getReportProgressCallback(pb);

        // std::cout << "SamplePoint 0 = {\n";
        // std::cout << "  pt: " << samplePoints[0].pt.transpose() << "\n";
        // std::cout << "  normal: " << samplePoints[0].normal.transpose() <<
        // "\n"; std::cout << "  directionForDerivative: " <<
        // samplePoints[0].directionForDerivative.transpose() << "\n"; std::cout
        // << "  type: " << static_cast<int>(samplePoints[0].type) << "\n";
        // std::cout << "  estimationQuantity: " <<
        // static_cast<int>(samplePoints[0].estimationQuantity) << "\n";
        // std::cout << "  pdf: " << samplePoints[0].pdf << "\n";
        // std::cout << "  distToAbsorbingBoundary: " <<
        // samplePoints[0].distToAbsorbingBoundary << "\n"; std::cout << "
        // distToReflectingBoundary: " <<
        // samplePoints[0].distToReflectingBoundary << "\n"; std::cout << "
        // firstSphereRadius: " << samplePoints[0].firstSphereRadius << "\n";
        // std::cout << "  robinCoeff: " << samplePoints[0].robinCoeff << "\n";
        // std::cout << "  solution: " << samplePoints[0].solution << "\n";
        // std::cout << "  normalDerivative: " <<
        // samplePoints[0].normalDerivative << "\n"; std::cout << "
        // contribution: " << samplePoints[0].contribution << "\n"; std::cout <<
        // "  estimateBoundaryNormalAligned: " << std::boolalpha <<
        // samplePoints[0].estimateBoundaryNormalAligned << "\n"; std::cout <<
        // "}\n";
        static constexpr size_t MAX_SIZE_FOR_TESTING = 1;
        if (samplePoints.size() > MAX_SIZE_FOR_TESTING) {
            std::vector<zombie::SamplePoint<float, DIM>> filteredPoints;
            filteredPoints.reserve(MAX_SIZE_FOR_TESTING);

            for (const auto &sp : samplePoints) {
                if (sp.type != zombie::SampleType::OnAbsorbingBoundary) {
                    filteredPoints.push_back(sp);
                    if (filteredPoints.size() == MAX_SIZE_FOR_TESTING)
                        break;
                }
            }

            if (filteredPoints.size() != MAX_SIZE_FOR_TESTING) {
                std::cerr << "Failed to find enough points off the absorbing "
                             "boundary\n";
                exit(-1);
            }

            samplePoints = std::move(filteredPoints);
            nWalks.resize(MAX_SIZE_FOR_TESTING);
        }

        SamplePoint *bonsai_pts =
            (SamplePoint *)malloc(sizeof(SamplePoint) * samplePoints.size());
        if constexpr (DIM == 3) {
            for (uint64_t i = 0; i < samplePoints.size(); i++) {
                bonsai_pts[i].pt = {samplePoints[i].pt(0),
                                    samplePoints[i].pt(1),
                                    samplePoints[i].pt(2)};
                bonsai_pts[i].normal = {samplePoints[i].normal(0),
                                        samplePoints[i].normal(1),
                                        samplePoints[i].normal(2)};
                bonsai_pts[i].pdf = samplePoints[i].pdf;
                bonsai_pts[i].distToAbs =
                    samplePoints[i].distToAbsorbingBoundary;
                bonsai_pts[i].distToRefl =
                    samplePoints[i].distToReflectingBoundary;
                bonsai_pts[i].type_and_quantity = 0;
                if (samplePoints[i].type == zombie::SampleType::InDomain) {
                    bonsai_pts[i].type_and_quantity |= 0;
                } else if (samplePoints[i].type ==
                           zombie::SampleType::OnAbsorbingBoundary) {
                    bonsai_pts[i].type_and_quantity |= 1;
                } else if (samplePoints[i].type ==
                           zombie::SampleType::OnReflectingBoundary) {
                    bonsai_pts[i].type_and_quantity |= 2;
                }

                if (samplePoints[i].estimationQuantity ==
                    zombie::EstimationQuantity::Solution) {
                    bonsai_pts[i].type_and_quantity |= 0;
                } else if (samplePoints[i].estimationQuantity ==
                           zombie::EstimationQuantity::SolutionAndGradient) {
                    bonsai_pts[i].type_and_quantity |= 4;
                } else if (samplePoints[i].estimationQuantity ==
                           zombie::EstimationQuantity::None) {
                    bonsai_pts[i].type_and_quantity |= 8;
                }

                if (samplePoints[i].estimateBoundaryNormalAligned) {
                    bonsai_pts[i].type_and_quantity |= 16;
                }
            }
        }

        // TODO(ajr): THIS IS WHAT WE NEED.
        zombie::WalkSettings walkSettings(
            epsilonShell, 0.0f, 0.0f, russianRouletteThreshold,
            splittingThreshold, maxWalkLength, stepsBeforeApplyingTikhonov,
            maxWalkLength, solveDoubleSided, !disableGradientControlVariates,
            !disableGradientAntitheticVariates,
            useCosineSamplingForDirectionalDerivatives,
            ignoreDirichletContribution, false, ignoreSourceContribution,
            printLogs);
        std::cout << "nWalks: " << nWalks[0] << std::endl;
        std::cout << "maxWalkLength: " << maxWalkLength << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        walkOnSpheres.solve(pde, walkSettings, nWalks, samplePoints,
                            runSingleThreaded, reportProgress);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "walkOnSpheres.solve took " << elapsed.count()
                  << " seconds.\n";
        pb.finish();

        if (false) {
            std::cout << "SamplePoint 0 = {\n";
            std::cout << "  pt: " << samplePoints[0].pt.transpose() << "\n";
            std::cout << "  normal: " << samplePoints[0].normal.transpose()
                      << "\n";
            std::cout << "  directionForDerivative: "
                      << samplePoints[0].directionForDerivative.transpose()
                      << "\n";
            std::cout << "  type: " << static_cast<int>(samplePoints[0].type)
                      << "\n";
            std::cout << "  estimationQuantity: "
                      << static_cast<int>(samplePoints[0].estimationQuantity)
                      << "\n";
            std::cout << "  pdf: " << samplePoints[0].pdf << "\n";
            std::cout << "  distToAbsorbingBoundary: "
                      << samplePoints[0].distToAbsorbingBoundary << "\n";
            std::cout << "  distToReflectingBoundary: "
                      << samplePoints[0].distToReflectingBoundary << "\n";
            std::cout << "  firstSphereRadius: "
                      << samplePoints[0].firstSphereRadius << "\n";
            std::cout << "  robinCoeff: " << samplePoints[0].robinCoeff << "\n";
            std::cout << "  solution: " << samplePoints[0].solution << "\n";
            std::cout << "  normalDerivative: "
                      << samplePoints[0].normalDerivative << "\n";
            std::cout << "  contribution: " << samplePoints[0].contribution
                      << "\n";
            std::cout << "  estimateBoundaryNormalAligned: " << std::boolalpha
                      << samplePoints[0].estimateBoundaryNormalAligned << "\n";
            std::cout << "}\n";
        }

        if (false) {
            const auto &stats = samplePoints[0].statistics;

            std::cout << "statistics.solutionMean = " << stats.solutionMean
                      << "\n";
            std::cout << "statistics.solutionM2 = " << stats.solutionM2 << "\n";

            std::cout << "statistics.gradientMean = [";
            for (size_t i = 0; i < DIM; ++i) {
                std::cout << stats.gradientMean[i];
                if (i + 1 < DIM)
                    std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "statistics.gradientM2 = [";
            for (size_t i = 0; i < DIM; ++i) {
                std::cout << stats.gradientM2[i];
                if (i + 1 < DIM)
                    std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "statistics.totalFirstSourceContribution = "
                      << stats.totalFirstSourceContribution << "\n";
            std::cout << "statistics.totalDerivativeContribution = "
                      << stats.totalDerivativeContribution << "\n";
            std::cout << "statistics.nSolutionEstimates = "
                      << stats.nSolutionEstimates << "\n";
            std::cout << "statistics.nGradientEstimates = "
                      << stats.nGradientEstimates << "\n";
            std::cout << "statistics.totalWalkLength = "
                      << stats.totalWalkLength << "\n";
            std::cout << "statistics.totalSplits = " << stats.totalSplits
                      << "\n";
        }

        // Now time to get my solver going.
        Statistics *bonsai_solution = nullptr;
        // constexpr uint64_t bonsai_fixed_count = 200;

        if constexpr (DIM == 3) {
            WalkSettings bonsai_ws;
            bonsai_ws.box = box;
            bonsai_ws.epsShellAbs = epsilonShell;
            bonsai_ws.epsShellRefl = 0.0; // TODO: not used
            bonsai_ws.silPrecision = 0.0; // TODO: not used
            bonsai_ws.russianRouletteThreshold = russianRouletteThreshold;
            bonsai_ws.maxWalkLength = maxWalkLength;
            bonsai_ws.stepsBeforeApplyingTikhonov =
                stepsBeforeApplyingTikhonov; // TODO: not used
            bonsai_ws.flags =
                (int)solveDoubleSided |
                (((int)!disableGradientControlVariates) << 1) |
                (((int)!disableGradientAntitheticVariates) << 2) |
                (((int)useCosineSamplingForDirectionalDerivatives) << 3) |
                (((int)ignoreDirichletContribution) << 4) |
                // ignoreReflectingBoundaryCondition is always false?
                (((int)ignoreSourceContribution) << 6) |
                (((int)printLogs) << 7);
            PDE bonsai_pde;
            bonsai_pde.absCoeff = pde.absorptionCoeff;
            bonsai_pde.freq = pde.freq;

            const uint32_t nSamplePts = samplePoints.size();
            // const uint64_t nSamplePts = bonsai_fixed_count;
            auto start = std::chrono::high_resolution_clock::now();
            std::cout << nSamplePts << "pts" << std::endl;

            std::cout << nWalksForSamplePts << " walks per pt" << std::endl;

            bonsai_solution = solve(bonsai_pde, bonsai_ws, nSamplePts,
                                    bonsai_pts, nWalksForSamplePts, tree);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "bonsai.solve took " << elapsed.count()
                      << " seconds.\n";

            if (false) {
                const Statistics &first = bonsai_solution[0];
                std::cout << "bonsai solutionMean = " << first.solMean << "\n";
                std::cout << "bonsai solutionM2 = " << first.solMean2 << "\n";
                std::cout << "bonsai nSolutionEstimates = "
                          << first.nSolEstimates << "\n";
                std::cout << "bonsai totalWalkLength = "
                          << first.totalWalkLength << "\n";
                std::cout << "bonsai totalSplits = " << first.totalSplits
                          << "\n";
                std::cout << "bonsai firstSphereRadius = "
                          << first.firstSphereRadius << "\n";
            }
        }

        // for (size_t i = 0; i <  samplePoints.size(); i++) {
        //     std::cout << "-----------------\ni = " << i << std::endl;
        //     std::cout << "zombie: " << samplePoints[i] << std::endl;
        //     std::cout << "bonsai: " << bonsai_pts[i] << std::endl;
        //     std::cout << bonsai_solution[i] << std::endl;
        //     std::cout << "-----------------\n";
        // }

        // plot results
        std::vector<float> firstSphereRadii;
        for (int i = 0; i < (int)samplePoints.size(); i++) {
            if (bonsai_solution) {
                firstSphereRadii.push_back(
                    bonsai_solution[i].firstSphereRadius);
            } else {
                firstSphereRadii.push_back(samplePoints[i].firstSphereRadius);
            }

            if (false) {
                // if (bonsai_solution && i < bonsai_fixed_count) {
                // TODO: assert within epsilon?
                const bool equal =
                    (samplePoints[i].firstSphereRadius ==
                     bonsai_solution[i].firstSphereRadius) &&
                    (samplePoints[i].statistics.solutionMean ==
                     bonsai_solution[i].solMean) &&
                    (samplePoints[i].statistics.solutionM2 ==
                     bonsai_solution[i].solMean2) &&
                    (samplePoints[i].statistics.totalFirstSourceContribution ==
                     bonsai_solution[i].totalFirstSourceContribution) &&
                    (samplePoints[i].statistics.nSolutionEstimates ==
                     bonsai_solution[i].nSolEstimates) &&
                    (samplePoints[i].statistics.totalWalkLength ==
                     bonsai_solution[i].totalWalkLength) &&
                    (samplePoints[i].statistics.totalSplits ==
                     bonsai_solution[i].totalSplits) &&
                    true;

                if (!equal) {
                    std::cerr << "Not equal at: i = " << i << std::endl;

                    if (samplePoints[i].firstSphereRadius !=
                        bonsai_solution[i].firstSphereRadius)
                        std::cerr << " firstSphereRadius: sample="
                                  << samplePoints[i].firstSphereRadius
                                  << ", bonsai="
                                  << bonsai_solution[i].firstSphereRadius
                                  << std::endl;

                    if (samplePoints[i].statistics.solutionMean !=
                        bonsai_solution[i].solMean)
                        std::cerr << " solutionMean: sample="
                                  << samplePoints[i].statistics.solutionMean
                                  << ", bonsai=" << bonsai_solution[i].solMean
                                  << std::endl;

                    if (samplePoints[i].statistics.solutionM2 !=
                        bonsai_solution[i].solMean2)
                        std::cerr << " solutionM2: sample="
                                  << samplePoints[i].statistics.solutionM2
                                  << ", bonsai=" << bonsai_solution[i].solMean2
                                  << std::endl;

                    if (samplePoints[i]
                            .statistics.totalFirstSourceContribution !=
                        bonsai_solution[i].totalFirstSourceContribution)
                        std::cerr
                            << " totalFirstSourceContribution: sample="
                            << samplePoints[i]
                                   .statistics.totalFirstSourceContribution
                            << ", bonsai="
                            << bonsai_solution[i].totalFirstSourceContribution
                            << std::endl;

                    if (samplePoints[i].statistics.nSolutionEstimates !=
                        bonsai_solution[i].nSolEstimates)
                        std::cerr
                            << " nSolutionEstimates: sample="
                            << samplePoints[i].statistics.nSolutionEstimates
                            << ", bonsai=" << bonsai_solution[i].nSolEstimates
                            << std::endl;

                    if (samplePoints[i].statistics.totalWalkLength !=
                        bonsai_solution[i].totalWalkLength)
                        std::cerr
                            << " totalWalkLength: sample="
                            << samplePoints[i].statistics.totalWalkLength
                            << ", bonsai=" << bonsai_solution[i].totalWalkLength
                            << std::endl;

                    if (samplePoints[i].statistics.totalSplits !=
                        bonsai_solution[i].totalSplits)
                        std::cerr
                            << " totalSplits: sample="
                            << samplePoints[i].statistics.totalSplits
                            << ", bonsai=" << bonsai_solution[i].totalSplits
                            << std::endl;
                    exit(-1);
                }
            }
        }

        auto pointCloud = polyscope::getPointCloud("Sample Points");
        pointCloud->addScalarQuantity("First Sphere Radii", firstSphereRadii);
        pointCloud->setPointRadiusQuantity("First Sphere Radii");

        // mask out sample values close to the boundary
        maskWalkOnSpheresEstimates(boundaryDistanceMask, samplePoints);

        // plot sample values
        plotWalkOnSpheresEstimates<DIM>(samplePoints, estimateOnSlicePlane,
                                        walkOnSpheresSolution);

        if (bonsai_solution) {
            // mask out sample values close to the boundary
            maskWalkOnSpheresEstimatesBonsai(boundaryDistanceMask, samplePoints,
                                             bonsai_solution);

            // plot sample values
            plotWalkOnSpheresEstimatesBonsai(
                bonsai_solution, samplePoints.size(), estimateOnSlicePlane,
                walkOnSpheresSolution);
        }

#ifdef USE_FEM
        if (!estimateOnSlicePlane) {
            plotAbsoluteDifference<DIM>(samplePoints.size(),
                                        walkOnSpheresSolution, femSolution);
        }
#endif
        nWalksTakenForSamplePts += nWalksForSamplePts;
        // }

        const std::string walksTakenPerSamplePtsStr =
            "Walks Taken Per Sample Point: " +
            std::to_string(nWalksTakenForSamplePts);
        exit(-1);
        ImGui::TextUnformatted(walksTakenPerSamplePtsStr.c_str());

        ImGui::Unindent();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef USE_FEM
    if (!domainIsOpen) {
        ImGui::TextUnformatted("FEM");
        ImGui::Indent();

        // solve with FEM
        ImGui::SliderFloat("Refinement Multiplier",
                           &femMeshRefinementMultiplier, 1.0f, 100.0f, "%.1f",
                           2.5f);
        if (ImGui::Button("Solve with FEM")) {
            std::vector<float> solutionRefined;
            std::vector<std::vector<float>> gradientRefined;
            femSolver.buildMesh(femMeshRefinementMultiplier);
            femSolver.solve(pde, solutionRefined, gradientRefined);

            std::vector<std::vector<float>> femGradient;
            std::vector<Vector<DIM>> meshPositionsRefined;
            std::vector<std::vector<size_t>> meshIndicesRefined;
            femSolver.extractMesh(meshPositionsRefined, meshIndicesRefined);
            femSolver.extractInterpolation(
                meshPositionsRefined, meshIndicesRefined, meshPositions,
                solutionRefined, gradientRefined, femSolution, femGradient);

            plotFemResults<DIM>(meshPositionsRefined, meshIndicesRefined,
                                solutionRefined, gradientRefined, femSolution,
                                femGradient);
            if (!estimateOnSlicePlane) {
                plotAbsoluteDifference<DIM>(samplePoints.size(),
                                            walkOnSpheresSolution, femSolution);
            }
        }

        ImGui::Unindent();
    }
#endif

    ImGui::PopItemWidth();
}

template <size_t DIM>
void runNoGUI(const std::vector<Vector<DIM>> &meshPositions,
              const std::vector<bool> &isBoundaryVertex,
              const zombie::GeometricQueries<DIM> &geometricQueries,
              const zombie::WalkOnSpheres<float, DIM> &walkOnSpheres,
#ifdef USE_FEM
              zombie::FemSolver<DIM> &femSolver,
#endif
              zombie::PDE<float, DIM> &pde, std::vector<int> &nWalks,
              std::vector<zombie::SamplePoint<float, DIM>> &samplePoints,
              std::vector<float> &walkOnSpheresSolution
#ifdef USE_FEM
              ,
              std::vector<float> &femSolution
#endif
) {

    if (!domainIsOpen) {
        if (estimateOnSlicePlane) {
            // initialize sample points as slice plane positions
            if (polyscope::state::slicePlanes.size() == 0) {
                polyscope::SlicePlane *psPlane =
                    polyscope::addSceneSlicePlane();
                psPlane->setDrawPlane(false);
                psPlane->setDrawWidget(false);
                psPlane->setActive(false);
                psPlane->setPose(glm::vec3{0., 0., 0.}, glm::vec3{0., 0, 1.});
            }

            std::vector<Vector<DIM>> slicePlanePositions;
            std::vector<std::vector<size_t>> slicePlaneIndices;
            glm::mat4 currentTransform =
                polyscope::state::slicePlanes[0]->getTransform();
            addSlicePlane<DIM>(geometricQueries.domainMin,
                               geometricQueries.domainMax, currentTransform,
                               slicePlaneResolution, slicePlanePositions,
                               slicePlaneIndices);
            addSlicePlaneSampleAndEvaluationPoints<DIM>(
                slicePlanePositions, slicePlaneIndices, geometricQueries,
                nWalksForSamplePts, estimateGradients, nWalks, samplePoints);

            if (DIM == 2) {
                polyscope::SurfaceMesh *mesh =
                    polyscope::getSurfaceMesh("Mesh");
                mesh->removeQuantity("Solution Estimate");
                mesh->removeQuantity("Solution Variance");
                mesh->removeQuantity("Gradient Estimate");
                mesh->removeQuantity("Mean Walk Length");
#ifdef USE_FEM
                mesh->removeQuantity("Absolute Difference");
#endif

            } else if (DIM == 3) {
                polyscope::VolumeMesh *mesh = polyscope::getVolumeMesh("Mesh");
                mesh->removeQuantity("Solution Estimate");
                mesh->removeQuantity("Solution Variance");
                mesh->removeQuantity("Gradient Estimate");
                mesh->removeQuantity("Mean Walk Length");
#ifdef USE_FEM
                mesh->removeQuantity("Absolute Difference");
#endif
            }

        } else {
            // initialize sample points as mesh positions
            addMeshSampleAndEvaluationPoints<DIM>(
                meshPositions, isBoundaryVertex, geometricQueries,
                nWalksForSamplePts, estimateGradients, nWalks, samplePoints);

            polyscope::removeStructure("Slice Plane", false);
            polyscope::removeLastSceneSlicePlane();
        }

        // plotSamplePoints<DIM>(samplePoints);

        nWalksTakenForSamplePts = 0;
    }
    // ImGui::Checkbox("Disable Gradient Control Variates",
    // &disableGradientControlVariates); ImGui::Checkbox("Disable Gradient
    // Antithetic Variates", &disableGradientAntitheticVariates);
    // ImGui::Checkbox("Use Cosine Sampling For Directional Derivatives",
    // &useCosineSamplingForDirectionalDerivatives); ImGui::Checkbox("Run Single
    // Threaded", &runSingleThreaded); ImGui::Checkbox("Print Logs",
    // &printLogs);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (!estimateOnSlicePlane || (estimateOnSlicePlane && solveDoubleSided)) {
        // ImGui::TextUnformatted("Point Estimation");
        // ImGui::Indent();

        // if (ImGui::SliderInt("Walks Per Sample Point", &nWalksForSamplePts,
        // 1, 10000)) {
        //     std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // }
        nWalksForSamplePts = 1;
        std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // if (ImGui::SliderInt("Walks Per Sample Point", &nWalksForSamplePts,
        // 1, 10000)) {
        //     std::fill(nWalks.begin(), nWalks.end(), nWalksForSamplePts);
        // }
        if (estimateGradients) {
            for (int i = 0; i < (int)samplePoints.size(); i++) {
                samplePoints[i].estimationQuantity = getEstimationQuantity(
                    samplePoints[i].type, estimateGradients);
            }

            clearResults<DIM>(samplePoints);
        }

        // solve with walk on spheres
        // if (ImGui::Button("Solve with Walk On Spheres")) {
        // estimate pointwise
        ProgressBar pb(samplePoints.size());
        std::function<void(int, int)> reportProgress =
            getReportProgressCallback(pb);

        // std::cout << "SamplePoint 0 = {\n";
        // std::cout << "  pt: " << samplePoints[0].pt.transpose() << "\n";
        // std::cout << "  normal: " << samplePoints[0].normal.transpose() <<
        // "\n"; std::cout << "  directionForDerivative: " <<
        // samplePoints[0].directionForDerivative.transpose() << "\n"; std::cout
        // << "  type: " << static_cast<int>(samplePoints[0].type) << "\n";
        // std::cout << "  estimationQuantity: " <<
        // static_cast<int>(samplePoints[0].estimationQuantity) << "\n";
        // std::cout << "  pdf: " << samplePoints[0].pdf << "\n";
        // std::cout << "  distToAbsorbingBoundary: " <<
        // samplePoints[0].distToAbsorbingBoundary << "\n"; std::cout << "
        // distToReflectingBoundary: " <<
        // samplePoints[0].distToReflectingBoundary << "\n"; std::cout << "
        // firstSphereRadius: " << samplePoints[0].firstSphereRadius << "\n";
        // std::cout << "  robinCoeff: " << samplePoints[0].robinCoeff << "\n";
        // std::cout << "  solution: " << samplePoints[0].solution << "\n";
        // std::cout << "  normalDerivative: " <<
        // samplePoints[0].normalDerivative << "\n"; std::cout << "
        // contribution: " << samplePoints[0].contribution << "\n"; std::cout <<
        // "  estimateBoundaryNormalAligned: " << std::boolalpha <<
        // samplePoints[0].estimateBoundaryNormalAligned << "\n"; std::cout <<
        // "}\n"; static constexpr size_t MAX_SIZE_FOR_TESTING = 10; if
        // (samplePoints.size() >  MAX_SIZE_FOR_TESTING) {
        //     std::vector<zombie::SamplePoint<float, DIM>> filteredPoints;
        //     filteredPoints.reserve(MAX_SIZE_FOR_TESTING);

        //     for (const auto& sp : samplePoints) {
        //         if (sp.type != zombie::SampleType::OnAbsorbingBoundary) {
        //             filteredPoints.push_back(sp);
        //             if (filteredPoints.size() == MAX_SIZE_FOR_TESTING)
        //                 break;
        //         }
        //     }

        //     if (filteredPoints.size() != MAX_SIZE_FOR_TESTING) {
        //         std::cerr << "Failed to find enough points off the absorbing
        //         boundary\n"; exit(-1);
        //     }

        //     samplePoints = std::move(filteredPoints);
        //     nWalks.resize(MAX_SIZE_FOR_TESTING);
        // }

        SamplePoint *bonsai_pts =
            (SamplePoint *)malloc(sizeof(SamplePoint) * samplePoints.size());
        if constexpr (DIM == 3) {
            for (uint64_t i = 0; i < samplePoints.size(); i++) {
                bonsai_pts[i].pt = {samplePoints[i].pt(0),
                                    samplePoints[i].pt(1),
                                    samplePoints[i].pt(2)};
                bonsai_pts[i].normal = {samplePoints[i].normal(0),
                                        samplePoints[i].normal(1),
                                        samplePoints[i].normal(2)};
                bonsai_pts[i].pdf = samplePoints[i].pdf;
                bonsai_pts[i].distToAbs =
                    samplePoints[i].distToAbsorbingBoundary;
                bonsai_pts[i].distToRefl =
                    samplePoints[i].distToReflectingBoundary;
                bonsai_pts[i].type_and_quantity = 0;
                if (samplePoints[i].type == zombie::SampleType::InDomain) {
                    bonsai_pts[i].type_and_quantity |= 0;
                } else if (samplePoints[i].type ==
                           zombie::SampleType::OnAbsorbingBoundary) {
                    bonsai_pts[i].type_and_quantity |= 1;
                } else if (samplePoints[i].type ==
                           zombie::SampleType::OnReflectingBoundary) {
                    bonsai_pts[i].type_and_quantity |= 2;
                }

                if (samplePoints[i].estimationQuantity ==
                    zombie::EstimationQuantity::Solution) {
                    bonsai_pts[i].type_and_quantity |= 0;
                } else if (samplePoints[i].estimationQuantity ==
                           zombie::EstimationQuantity::SolutionAndGradient) {
                    bonsai_pts[i].type_and_quantity |= 4;
                } else if (samplePoints[i].estimationQuantity ==
                           zombie::EstimationQuantity::None) {
                    bonsai_pts[i].type_and_quantity |= 8;
                }

                if (samplePoints[i].estimateBoundaryNormalAligned) {
                    bonsai_pts[i].type_and_quantity |= 16;
                }
            }
        }

        // TODO(ajr): THIS IS WHAT WE NEED.
        zombie::WalkSettings walkSettings(
            epsilonShell, 0.0f, 0.0f, russianRouletteThreshold,
            splittingThreshold, maxWalkLength, stepsBeforeApplyingTikhonov,
            maxWalkLength, solveDoubleSided, !disableGradientControlVariates,
            !disableGradientAntitheticVariates,
            useCosineSamplingForDirectionalDerivatives,
            ignoreDirichletContribution, false, ignoreSourceContribution,
            printLogs);
        std::cout << "nWalks: " << nWalks[0] << std::endl;
        std::cout << "maxWalkLength: " << maxWalkLength << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        walkOnSpheres.solve(pde, walkSettings, nWalks, samplePoints,
                            runSingleThreaded, reportProgress);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "walkOnSpheres.solve took " << elapsed.count()
                  << " seconds.\n";
        pb.finish();

        if (false) {
            std::cout << "SamplePoint 0 = {\n";
            std::cout << "  pt: " << samplePoints[0].pt.transpose() << "\n";
            std::cout << "  normal: " << samplePoints[0].normal.transpose()
                      << "\n";
            std::cout << "  directionForDerivative: "
                      << samplePoints[0].directionForDerivative.transpose()
                      << "\n";
            std::cout << "  type: " << static_cast<int>(samplePoints[0].type)
                      << "\n";
            std::cout << "  estimationQuantity: "
                      << static_cast<int>(samplePoints[0].estimationQuantity)
                      << "\n";
            std::cout << "  pdf: " << samplePoints[0].pdf << "\n";
            std::cout << "  distToAbsorbingBoundary: "
                      << samplePoints[0].distToAbsorbingBoundary << "\n";
            std::cout << "  distToReflectingBoundary: "
                      << samplePoints[0].distToReflectingBoundary << "\n";
            std::cout << "  firstSphereRadius: "
                      << samplePoints[0].firstSphereRadius << "\n";
            std::cout << "  robinCoeff: " << samplePoints[0].robinCoeff << "\n";
            std::cout << "  solution: " << samplePoints[0].solution << "\n";
            std::cout << "  normalDerivative: "
                      << samplePoints[0].normalDerivative << "\n";
            std::cout << "  contribution: " << samplePoints[0].contribution
                      << "\n";
            std::cout << "  estimateBoundaryNormalAligned: " << std::boolalpha
                      << samplePoints[0].estimateBoundaryNormalAligned << "\n";
            std::cout << "}\n";
        }

        if (false) {
            const auto &stats = samplePoints[0].statistics;

            std::cout << "statistics.solutionMean = " << stats.solutionMean
                      << "\n";
            std::cout << "statistics.solutionM2 = " << stats.solutionM2 << "\n";

            std::cout << "statistics.gradientMean = [";
            for (size_t i = 0; i < DIM; ++i) {
                std::cout << stats.gradientMean[i];
                if (i + 1 < DIM)
                    std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "statistics.gradientM2 = [";
            for (size_t i = 0; i < DIM; ++i) {
                std::cout << stats.gradientM2[i];
                if (i + 1 < DIM)
                    std::cout << ", ";
            }
            std::cout << "]\n";

            std::cout << "statistics.totalFirstSourceContribution = "
                      << stats.totalFirstSourceContribution << "\n";
            std::cout << "statistics.totalDerivativeContribution = "
                      << stats.totalDerivativeContribution << "\n";
            std::cout << "statistics.nSolutionEstimates = "
                      << stats.nSolutionEstimates << "\n";
            std::cout << "statistics.nGradientEstimates = "
                      << stats.nGradientEstimates << "\n";
            std::cout << "statistics.totalWalkLength = "
                      << stats.totalWalkLength << "\n";
            std::cout << "statistics.totalSplits = " << stats.totalSplits
                      << "\n";
        }

        // Now time to get my solver going.
        Statistics *bonsai_solution = nullptr;
        // constexpr uint64_t bonsai_fixed_count = 200;

        if constexpr (DIM == 3) {
            WalkSettings bonsai_ws;
            bonsai_ws.box = box;
            bonsai_ws.epsShellAbs = epsilonShell;
            bonsai_ws.epsShellRefl = 0.0; // TODO: not used
            bonsai_ws.silPrecision = 0.0; // TODO: not used
            bonsai_ws.russianRouletteThreshold = russianRouletteThreshold;
            bonsai_ws.maxWalkLength = maxWalkLength;
            bonsai_ws.stepsBeforeApplyingTikhonov =
                stepsBeforeApplyingTikhonov; // TODO: not used
            bonsai_ws.flags =
                (int)solveDoubleSided |
                (((int)!disableGradientControlVariates) << 1) |
                (((int)!disableGradientAntitheticVariates) << 2) |
                (((int)useCosineSamplingForDirectionalDerivatives) << 3) |
                (((int)ignoreDirichletContribution) << 4) |
                // ignoreReflectingBoundaryCondition is always false?
                (((int)ignoreSourceContribution) << 6) |
                (((int)printLogs) << 7);
            PDE bonsai_pde;
            bonsai_pde.absCoeff = pde.absorptionCoeff;
            bonsai_pde.freq = pde.freq;

            const uint32_t nSamplePts = samplePoints.size();
            // const uint64_t nSamplePts = bonsai_fixed_count;
            auto start = std::chrono::high_resolution_clock::now();
            std::cout << nSamplePts << "pts" << std::endl;

            std::cout << nWalksForSamplePts << " walks per pt" << std::endl;

            bonsai_solution = solve(bonsai_pde, bonsai_ws, nSamplePts,
                                    bonsai_pts, nWalksForSamplePts, tree);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "bonsai.solve took " << elapsed.count()
                      << " seconds.\n";

            if (false) {
                const Statistics &first = bonsai_solution[0];
                std::cout << "bonsai solutionMean = " << first.solMean << "\n";
                std::cout << "bonsai solutionM2 = " << first.solMean2 << "\n";
                std::cout << "bonsai nSolutionEstimates = "
                          << first.nSolEstimates << "\n";
                std::cout << "bonsai totalWalkLength = "
                          << first.totalWalkLength << "\n";
                std::cout << "bonsai totalSplits = " << first.totalSplits
                          << "\n";
                std::cout << "bonsai firstSphereRadius = "
                          << first.firstSphereRadius << "\n";
            }
        }

        // for (size_t i = 0; i <  samplePoints.size(); i++) {
        //     std::cout << "-----------------\ni = " << i << std::endl;
        //     std::cout << "zombie: " << samplePoints[i] << std::endl;
        //     std::cout << "bonsai: " << bonsai_pts[i] << std::endl;
        //     std::cout << bonsai_solution[i] << std::endl;
        //     std::cout << "-----------------\n";
        // }

        // plot results
        std::vector<float> firstSphereRadii;
        for (int i = 0; i < (int)samplePoints.size(); i++) {
            if (bonsai_solution) {
                firstSphereRadii.push_back(
                    bonsai_solution[i].firstSphereRadius);
            } else {
                firstSphereRadii.push_back(samplePoints[i].firstSphereRadius);
            }

            if (false) {
                // if (bonsai_solution && i < bonsai_fixed_count) {
                // TODO: assert within epsilon?
                const bool equal =
                    (samplePoints[i].firstSphereRadius ==
                     bonsai_solution[i].firstSphereRadius) &&
                    (samplePoints[i].statistics.solutionMean ==
                     bonsai_solution[i].solMean) &&
                    (samplePoints[i].statistics.solutionM2 ==
                     bonsai_solution[i].solMean2) &&
                    (samplePoints[i].statistics.totalFirstSourceContribution ==
                     bonsai_solution[i].totalFirstSourceContribution) &&
                    (samplePoints[i].statistics.nSolutionEstimates ==
                     bonsai_solution[i].nSolEstimates) &&
                    (samplePoints[i].statistics.totalWalkLength ==
                     bonsai_solution[i].totalWalkLength) &&
                    (samplePoints[i].statistics.totalSplits ==
                     bonsai_solution[i].totalSplits) &&
                    true;

                if (!equal) {
                    std::cerr << "Not equal at: i = " << i << std::endl;

                    if (samplePoints[i].firstSphereRadius !=
                        bonsai_solution[i].firstSphereRadius)
                        std::cerr << " firstSphereRadius: sample="
                                  << samplePoints[i].firstSphereRadius
                                  << ", bonsai="
                                  << bonsai_solution[i].firstSphereRadius
                                  << std::endl;

                    if (samplePoints[i].statistics.solutionMean !=
                        bonsai_solution[i].solMean)
                        std::cerr << " solutionMean: sample="
                                  << samplePoints[i].statistics.solutionMean
                                  << ", bonsai=" << bonsai_solution[i].solMean
                                  << std::endl;

                    if (samplePoints[i].statistics.solutionM2 !=
                        bonsai_solution[i].solMean2)
                        std::cerr << " solutionM2: sample="
                                  << samplePoints[i].statistics.solutionM2
                                  << ", bonsai=" << bonsai_solution[i].solMean2
                                  << std::endl;

                    if (samplePoints[i]
                            .statistics.totalFirstSourceContribution !=
                        bonsai_solution[i].totalFirstSourceContribution)
                        std::cerr
                            << " totalFirstSourceContribution: sample="
                            << samplePoints[i]
                                   .statistics.totalFirstSourceContribution
                            << ", bonsai="
                            << bonsai_solution[i].totalFirstSourceContribution
                            << std::endl;

                    if (samplePoints[i].statistics.nSolutionEstimates !=
                        bonsai_solution[i].nSolEstimates)
                        std::cerr
                            << " nSolutionEstimates: sample="
                            << samplePoints[i].statistics.nSolutionEstimates
                            << ", bonsai=" << bonsai_solution[i].nSolEstimates
                            << std::endl;

                    if (samplePoints[i].statistics.totalWalkLength !=
                        bonsai_solution[i].totalWalkLength)
                        std::cerr
                            << " totalWalkLength: sample="
                            << samplePoints[i].statistics.totalWalkLength
                            << ", bonsai=" << bonsai_solution[i].totalWalkLength
                            << std::endl;

                    if (samplePoints[i].statistics.totalSplits !=
                        bonsai_solution[i].totalSplits)
                        std::cerr
                            << " totalSplits: sample="
                            << samplePoints[i].statistics.totalSplits
                            << ", bonsai=" << bonsai_solution[i].totalSplits
                            << std::endl;
                    exit(-1);
                }
            }
        }

        auto pointCloud = polyscope::getPointCloud("Sample Points");
        pointCloud->addScalarQuantity("First Sphere Radii", firstSphereRadii);
        pointCloud->setPointRadiusQuantity("First Sphere Radii");

        // mask out sample values close to the boundary
        maskWalkOnSpheresEstimates(boundaryDistanceMask, samplePoints);

        // plot sample values
        plotWalkOnSpheresEstimates<DIM>(samplePoints, estimateOnSlicePlane,
                                        walkOnSpheresSolution);

        if (bonsai_solution) {
            // mask out sample values close to the boundary
            maskWalkOnSpheresEstimatesBonsai(boundaryDistanceMask, samplePoints,
                                             bonsai_solution);

            // plot sample values
            plotWalkOnSpheresEstimatesBonsai(
                bonsai_solution, samplePoints.size(), estimateOnSlicePlane,
                walkOnSpheresSolution);
        }

        nWalksTakenForSamplePts += nWalksForSamplePts;
        // }

        const std::string walksTakenPerSamplePtsStr =
            "Walks Taken Per Sample Point: " +
            std::to_string(nWalksTakenForSamplePts);
        exit(-1);
    }
}

template <size_t DIM>
void visualizeScene(const std::vector<Vector<DIM>> &meshPositions,
                    const std::vector<Vector<DIM>> &boundaryPositions,
                    const std::vector<std::vector<size_t>> &meshIndices,
                    const std::vector<Vectori<DIM>> &boundaryIndices,
                    const std::vector<bool> &isBoundaryVertex,
                    const zombie::GeometricQueries<DIM> &geometricQueries,
                    const zombie::WalkOnSpheres<float, DIM> &walkOnSpheres,
#ifdef USE_FEM
                    zombie::FemSolver<DIM> &femSolver,
#endif
                    zombie::PDE<float, DIM> &pde) {
    // set a few options
    polyscope::options::programName = "walk on spheres";
    polyscope::options::verbosity = 0;
    polyscope::options::usePrefsFile = false;
    polyscope::options::autocenterStructures = false;

    // initialize polyscope
    polyscope::init();

    // register meshes, curves and point clouds
    if (DIM == 2) {
        if (!domainIsOpen)
            polyscope::registerSurfaceMesh2D("Mesh", meshPositions,
                                             meshIndices);
        polyscope::registerCurveNetwork2D("Boundary", boundaryPositions,
                                          boundaryIndices);

    } else if (DIM == 3) {
        if (!domainIsOpen)
            polyscope::registerTetMesh("Mesh", meshPositions, meshIndices);
        polyscope::registerSurfaceMesh("Boundary", boundaryPositions,
                                       boundaryIndices);
    }

    // initialize sample and evaluation points
    std::vector<int> nWalks;
    std::vector<zombie::SamplePoint<float, DIM>> samplePoints;
    if (estimateOnSlicePlane) {
        // initialize sample points as slice plane positions
        if (polyscope::state::slicePlanes.size() == 0) {
            polyscope::SlicePlane *psPlane = polyscope::addSceneSlicePlane();
            psPlane->setDrawPlane(false);
            psPlane->setDrawWidget(false);
            psPlane->setActive(false);
            psPlane->setPose(glm::vec3{0., 0., 0.}, glm::vec3{0., 0, 1.});
        }

        std::vector<Vector<DIM>> slicePlanePositions;
        std::vector<std::vector<size_t>> slicePlaneIndices;
        glm::mat4 currentTransform =
            polyscope::state::slicePlanes[0]->getTransform();
        addSlicePlane<DIM>(geometricQueries.domainMin,
                           geometricQueries.domainMax, currentTransform,
                           slicePlaneResolution, slicePlanePositions,
                           slicePlaneIndices);
        addSlicePlaneSampleAndEvaluationPoints<DIM>(
            slicePlanePositions, slicePlaneIndices, geometricQueries,
            nWalksForSamplePts, estimateGradients, nWalks, samplePoints);

    } else {
        // initialize sample points as mesh positions
        addMeshSampleAndEvaluationPoints<DIM>(
            meshPositions, isBoundaryVertex, geometricQueries,
            nWalksForSamplePts, estimateGradients, nWalks, samplePoints);
    }

    plotSamplePoints<DIM>(samplePoints);

    // bind gui callback
    std::vector<float> walkOnSpheresSolution;
    // #ifdef USE_FEM
    //     std::vector<float> femSolution;
    //     polyscope::state::userCallback = std::bind(&guiCallback<DIM>,
    //     std::cref(meshPositions),
    //                                                std::cref(isBoundaryVertex),
    //                                                std::cref(geometricQueries),
    //                                                std::cref(walkOnSpheres),
    //                                                std::ref(femSolver),
    //                                                std::ref(pde),
    //                                                std::ref(nWalks),
    //                                                std::ref(samplePoints),
    //                                                std::ref(walkOnSpheresSolution),
    //                                                std::ref(femSolution));
    // #else
    //     polyscope::state::userCallback = std::bind(&guiCallback<DIM>,
    //     std::cref(meshPositions),
    //                                                std::cref(isBoundaryVertex),
    //                                                std::cref(geometricQueries),
    //                                                std::cref(walkOnSpheres),
    //                                                std::ref(pde),
    //                                                std::ref(nWalks),
    //                                                std::ref(samplePoints),
    //                                                std::ref(walkOnSpheresSolution));
    // #endif

    //     // give control to polyscope gui
    //     polyscope::show();
    runNoGUI<DIM>(std::cref(meshPositions), std::cref(isBoundaryVertex),
                  std::cref(geometricQueries), std::cref(walkOnSpheres),
                  std::ref(pde), std::ref(nWalks), std::ref(samplePoints),
                  std::ref(walkOnSpheresSolution));
}

template <size_t DIM>
void run() {
    // load geometry
    std::vector<bool> isBoundaryVertex;
    std::vector<Vector<DIM>> meshPositions, boundaryPositions;
    std::vector<std::vector<size_t>> meshIndices;
    std::vector<Vectori<DIM>> boundaryIndices;
    if (domainIsOpen) {
        zombie::loadBoundaryMesh<DIM>(filename, boundaryPositions,
                                      boundaryIndices);
        zombie::normalize<DIM>(boundaryPositions);
        if (flipMeshOrientation)
            zombie::flipOrientation<DIM>(boundaryIndices);

        meshPositions = boundaryPositions;
        for (size_t i = 0; i < boundaryIndices.size(); ++i) {
            meshIndices.emplace_back(std::vector<size_t>());
            for (size_t j = 0; j < boundaryIndices[i].size(); ++j) {
                meshIndices[i].emplace_back(boundaryIndices[i][j]);
            }
        }

    } else {
        loadVolumeMesh<DIM>(filename, flipMeshOrientation, meshPositions,
                            meshIndices, boundaryPositions, boundaryIndices,
                            isBoundaryVertex);
    }

    // compute bounding boxes
    std::pair<Vector<DIM>, Vector<DIM>> squareBoxExtents =
        zombie::computeBoundingBox<DIM>(meshPositions, true, 1.15f);
    std::pair<Vector<DIM>, Vector<DIM>> tightBoxExtents =
        zombie::computeBoundingBox<DIM>(meshPositions, false, 1.0f);
    auto tightBoxContains = [&tightBoxExtents](const Vector<DIM> &x) -> bool {
        const Vector<DIM> &bMin = tightBoxExtents.first;
        const Vector<DIM> &bMax = tightBoxExtents.second;

        return (x.array() >= bMin.array()).all() &&
               (x.array() <= bMax.array()).all();
    };
    if (solveDoubleSided) {
        // add square box to geometry
        zombie::addBoundingBoxToBoundaryMesh<DIM>(
            squareBoxExtents.first, squareBoxExtents.second, boundaryPositions,
            boundaryIndices);
    }

    // build boundary aggregate
    zombie::FcpwDirichletBoundaryHandler<DIM> boundaryHandler;
    boundaryHandler.buildAccelerationStructure(boundaryPositions,
                                               boundaryIndices);

    // TODO(ajr): also build Bonsai aggregate
    if constexpr (DIM == 3) {
        auto start = std::chrono::high_resolution_clock::now();
        fcpw::Aggregate<DIM> *aggregate =
            boundaryHandler.scene.getSceneData()->aggregate.get();
        if (auto bvh = dynamic_cast<fcpwTriBVH *>(aggregate)) {
            std::cout << "Converting FCPW tree (good)\n";
            tree = convert_tree(bvh);
        } else {
            std::cout << "Building Bonsai tree (bad)\n";
            tree = build_tree(boundaryPositions, boundaryIndices);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "bonsai.tree_build took " << elapsed.count()
                  << " seconds.\n";
    }

    // populate geometric queries
    zombie::GeometricQueries<DIM> geometricQueries(
        !domainIsOpen, squareBoxExtents.first, squareBoxExtents.second);
    zombie::populateGeometricQueriesForDirichletBoundary<DIM>(boundaryHandler,
                                                              geometricQueries);

    zombie::SdfGrid<DIM> sdfGrid(squareBoxExtents.first,
                                 squareBoxExtents.second);
    if (!solveDoubleSided && useSdfForBoundary) {
        // override distance queries to use an SDF grid. The user can also use
        // Zombie to build an SDF hierarchy for double-sided problems (ommited
        // here for simplicity)
        Vectori<DIM> sdfGridShape = Vectori<DIM>::Constant(sdfGridResolution);
        zombie::populateSdfGrid<DIM>(boundaryHandler, sdfGrid, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<
            zombie::SdfGrid<DIM>, DIM>(sdfGrid, geometricQueries);
    }

    // initialize walk on spheres solver
    zombie::WalkOnSpheres<float, DIM> walkOnSpheres(geometricQueries);

#ifdef USE_FEM
    // initialize the fem solver
    zombie::FemSolver<DIM> femSolver(meshPositions, meshIndices);
#endif

    // setup pde
    float boxExtent = (tightBoxExtents.second - tightBoxExtents.first).norm();
    float freq = 10.0f / boxExtent;
    zombie::PDE<float, DIM> pde;
    pde.source = [freq](const Vector<DIM> &x) -> float {
        return ::ignoreSourceContribution
                   ? 0.0f
                   : 5.0f * std::sin(freq * x.y()) * std::cos(freq * x.x());
    };
    pde.dirichlet = [freq, &tightBoxContains](
                        const Vector<DIM> &x,
                        bool returnBoundaryNormalAlignedValue) -> float {
        if (::solveDoubleSided && !tightBoxContains(x))
            return 0.0f;
        return ::ignoreDirichletContribution
                   ? 0.0f
                   : std::sin(freq * x.x()) * std::cos(freq * x.y());
    };
    pde.hasReflectingBoundaryConditions = [](const Vector<DIM> &x) -> bool {
        return false;
    };
    pde.absorptionCoeff = absorptionCoeff;
    pde.freq = 10.0f / boxExtent;

    const auto &eigenLow = tightBoxExtents.first;
    box.low[0] = eigenLow(0);
    box.low[1] = eigenLow(1);
    box.low[2] = eigenLow(2);

    const auto &eigenHigh = tightBoxExtents.second;
    box.high[0] = eigenHigh(0);
    box.high[1] = eigenHigh(1);
    box.high[2] = eigenHigh(2);

    // visualize the scene
    visualizeScene<DIM>(meshPositions, boundaryPositions, meshIndices,
                        boundaryIndices, isBoundaryVertex, geometricQueries,
                        walkOnSpheres,
#ifdef USE_FEM
                        femSolver,
#endif
                        pde);
}

int main(int argc, const char *argv[]) {
    // configure the argument parser
    args::ArgumentParser parser("walk on stars");
    args::Group group(parser, "", args::Group::Validators::DontCare);
    args::Flag flipMeshOrientation(group, "bool",
                                   "flip mesh orientation on load",
                                   {"flipMeshOrientation"});
    args::Flag domainIsOpen(group, "bool", "domain is open", {"domainIsOpen"});
    args::Flag solveDoubleSided(group, "bool", "solve double sided",
                                {"solveDoubleSided"});
    args::Flag estimateGradients(group, "bool", "estimate gradients",
                                 {"estimateGradients"});
    args::Flag disableGradientControlVariates(
        group, "bool", "disable gradient control variates",
        {"disableGradientControlVariates"});
    args::Flag disableGradientAntitheticVariates(
        group, "bool", "disable gradient antithetic variates",
        {"disableGradientAntitheticVariates"});
    args::Flag useCosineSamplingForDirectionalDerivatives(
        group, "bool", "use cosine sampling for directional derivatives",
        {"useCosineSamplingForDirectionalDerivatives"});
    args::Flag ignoreDirichletContribution(
        group, "bool", "ignore Dirichlet values during random walk",
        {"ignoreDirichletContribution"});
    args::Flag ignoreSourceContribution(
        group, "bool", "ignore source values during random walk",
        {"ignoreSourceContribution"});
    args::Flag runSingleThreaded(group, "bool", "run single threaded",
                                 {"runSingleThreaded"});
    args::Flag printLogs(group, "bool", "print logs", {"printLogs"});
    args::Flag useSdfForBoundary(group, "bool", "use SDF for boundary",
                                 {"useSdfForBoundary"});
    args::ValueFlag<std::string> filename(parser, "string",
                                          "polygon soup filename", {"file"});
    args::ValueFlag<int> dim(parser, "int", "scene dimension", {"dim"});
    args::ValueFlag<int> nWalksForSamplePts(
        parser, "int", "walks per point for point estimator",
        {"nWalksForSamplePts"});
    args::ValueFlag<int> maxWalkLength(
        parser, "int", "max walk length for walk on stars", {"maxWalkLength"});
    args::ValueFlag<int> stepsBeforeApplyingTikhonov(
        parser, "int", "steps before applying Tikhonov regularization",
        {"stepsBeforeApplyingTikhonov"});
    args::ValueFlag<int> slicePlaneResolution(
        parser, "int", "slice plane resolution", {"slicePlaneResolution"});
    args::ValueFlag<int> sdfGridResolution(parser, "int", "SDF grid resolution",
                                           {"sdfGridResolution"});
    args::ValueFlag<float> absorptionCoeff(
        parser, "float", "absorption coefficient for screened Poisson eqution",
        {"absorptionCoeff"});
    args::ValueFlag<float> epsilonShell(parser, "float", "epsilon shell",
                                        {"epsilonShell"});
    args::ValueFlag<float> russianRouletteThreshold(
        parser, "float", "russian roulette threshold",
        {"russianRouletteThreshold"});
    args::ValueFlag<float> splittingThreshold(
        parser, "float", "splitting threshold", {"splittingThreshold"});
    args::ValueFlag<float> boundaryDistanceMask(
        parser, "float", "boundary distance for masking out estimated values",
        {"boundaryDistanceMask"});
#ifdef USE_FEM
    args::ValueFlag<float> femMeshRefinementMultiplier(
        parser, "float", "fem mesh refinement multiplier",
        {"femMeshRefinementMultiplier"});
#endif

    // parse args
    try {
        parser.ParseCLI(argc, argv);

    } catch (const args::Help &) {
        std::cout << parser;
        return 0;

    } catch (const args::ParseError &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << parser;
        return 1;
    }

    int DIM = args::get(dim);
    if (!dim) {
        std::cerr << "Specify dimension" << std::endl;
        return EXIT_FAILURE;

    } else {
        if (DIM != 2 && DIM != 3) {
            std::cerr << "Only dimensions 2 and 3 are supported" << std::endl;
            return EXIT_FAILURE;
        }
    }

    if (!filename) {
        std::cerr << "Specify a filename" << std::endl;
        return EXIT_FAILURE;

    } else {
        ::filename = args::get(filename);
    }

    ::flipMeshOrientation = args::get(flipMeshOrientation);
    ::domainIsOpen = args::get(domainIsOpen);
    ::solveDoubleSided = args::get(solveDoubleSided) || args::get(domainIsOpen);
    ::estimateGradients = args::get(estimateGradients);
    ::disableGradientControlVariates =
        args::get(disableGradientControlVariates);
    ::disableGradientAntitheticVariates =
        args::get(disableGradientAntitheticVariates);
    ::useCosineSamplingForDirectionalDerivatives =
        args::get(useCosineSamplingForDirectionalDerivatives);
    ::ignoreDirichletContribution = args::get(ignoreDirichletContribution);
    ::ignoreSourceContribution = args::get(ignoreSourceContribution);
    ::runSingleThreaded = args::get(runSingleThreaded);
    ::printLogs = args::get(printLogs);
    ::estimateOnSlicePlane = args::get(domainIsOpen);
    ::useSdfForBoundary = args::get(useSdfForBoundary);
    if (nWalksForSamplePts)
        ::nWalksForSamplePts = args::get(nWalksForSamplePts);
    if (maxWalkLength)
        ::maxWalkLength = args::get(maxWalkLength);
    if (stepsBeforeApplyingTikhonov)
        ::stepsBeforeApplyingTikhonov = args::get(stepsBeforeApplyingTikhonov);
    if (slicePlaneResolution)
        ::slicePlaneResolution = args::get(slicePlaneResolution);
    if (sdfGridResolution)
        ::sdfGridResolution = args::get(sdfGridResolution);
    if (absorptionCoeff)
        ::absorptionCoeff = args::get(absorptionCoeff);
    if (epsilonShell)
        ::epsilonShell = args::get(epsilonShell);
    if (russianRouletteThreshold)
        ::russianRouletteThreshold = args::get(russianRouletteThreshold);
    if (splittingThreshold)
        ::splittingThreshold = args::get(splittingThreshold);
    if (boundaryDistanceMask)
        ::boundaryDistanceMask = args::get(boundaryDistanceMask);
#ifdef USE_FEM
    if (femMeshRefinementMultiplier)
        ::femMeshRefinementMultiplier = args::get(femMeshRefinementMultiplier);
#endif

    // run app
    if (DIM == 2)
        run<2>();
    else if (DIM == 3)
        run<3>();

    return 0;
}
