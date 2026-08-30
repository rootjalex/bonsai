// Driver for apps/pbrt/render.bonsai, milestone 0.
//
//   ./build/compiler -p ssa -b cpp -i apps/pbrt/render.bonsai -o apps/pbrt/render
//   clang++ -std=c++20 -O3 apps/pbrt/render_hook.cpp apps/pbrt/render.o \
//       -o apps/pbrt/render_runner
//   ./apps/pbrt/render_runner out.ppm
//
// Everything PBRT does in a camera constructor happens here: the field of
// view, the screen window and the resolution become two matrices, and the
// bonsai side only ever applies them. That split is the app's rule, not a
// shortcut -- bonsai has no file I/O to parse a .pbrt scene with, and PBRT's
// hot path does not build transforms either.

#include "render.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>

namespace {

constexpr float Pi = 3.14159265358979323846f;

float radians(float deg) { return deg * Pi / 180.0f; }

// pbrt: SquareMatrix<4>, with just the operations the camera needs.
struct Mat4 {
    float m[4][4];

    static Mat4 identity() {
        Mat4 r{};
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                r.m[i][j] = (i == j) ? 1.0f : 0.0f;
            }
        }
        return r;
    }
};

Mat4 operator*(const Mat4 &a, const Mat4 &b) {
    Mat4 r{};
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) {
                s += a.m[i][k] * b.m[k][j];
            }
            r.m[i][j] = s;
        }
    }
    return r;
}

// pbrt: Inverse(SquareMatrix<4>), by Gauss-Jordan with full pivoting. Written
// out rather than derived by hand for each transform, because the matrices
// below are compositions and inverting them individually is where the sign
// errors live.
Mat4 inverse(const Mat4 &in) {
    Mat4 a = in;
    Mat4 r = Mat4::identity();
    for (int col = 0; col < 4; col++) {
        int pivot = col;
        for (int row = col + 1; row < 4; row++) {
            if (std::fabs(a.m[row][col]) > std::fabs(a.m[pivot][col])) {
                pivot = row;
            }
        }
        if (a.m[pivot][col] == 0.0f) {
            std::cerr << "singular camera matrix\n";
            std::exit(1);
        }
        if (pivot != col) {
            for (int k = 0; k < 4; k++) {
                std::swap(a.m[pivot][k], a.m[col][k]);
                std::swap(r.m[pivot][k], r.m[col][k]);
            }
        }
        const float inv = 1.0f / a.m[col][col];
        for (int k = 0; k < 4; k++) {
            a.m[col][k] *= inv;
            r.m[col][k] *= inv;
        }
        for (int row = 0; row < 4; row++) {
            if (row == col) {
                continue;
            }
            const float f = a.m[row][col];
            for (int k = 0; k < 4; k++) {
                a.m[row][k] -= f * a.m[col][k];
                r.m[row][k] -= f * r.m[col][k];
            }
        }
    }
    return r;
}

Mat4 scale(float x, float y, float z) {
    Mat4 r = Mat4::identity();
    r.m[0][0] = x;
    r.m[1][1] = y;
    r.m[2][2] = z;
    return r;
}

Mat4 translate(float x, float y, float z) {
    Mat4 r = Mat4::identity();
    r.m[0][3] = x;
    r.m[1][3] = y;
    r.m[2][3] = z;
    return r;
}

// pbrt: Perspective(fov, n, f).
Mat4 perspective(float fov, float n, float f) {
    Mat4 persp = Mat4::identity();
    persp.m[2][2] = f / (f - n);
    persp.m[2][3] = -f * n / (f - n);
    persp.m[3][2] = 1.0f;
    persp.m[3][3] = 0.0f;
    const float inv_tan_ang = 1.0f / std::tan(radians(fov) / 2.0f);
    return scale(inv_tan_ang, inv_tan_ang, 1.0f) * persp;
}

struct Vec3 {
    float x, y, z;
};

Vec3 operator-(const Vec3 &a, const Vec3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Vec3 cross(const Vec3 &a, const Vec3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}

Vec3 normalize(const Vec3 &v) {
    const float n = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return {v.x / n, v.y / n, v.z / n};
}

// pbrt: LookAt(pos, look, up), which returns world-from-camera. PBRT then
// inverts it for its camera-from-world; M0 wants this direction as it is.
Mat4 look_at(const Vec3 &pos, const Vec3 &look, const Vec3 &up) {
    const Vec3 dir = normalize(look - pos);
    const Vec3 right = normalize(cross(normalize(up), dir));
    const Vec3 new_up = cross(dir, right);

    Mat4 r = Mat4::identity();
    r.m[0][0] = right.x; r.m[0][1] = new_up.x; r.m[0][2] = dir.x; r.m[0][3] = pos.x;
    r.m[1][0] = right.y; r.m[1][1] = new_up.y; r.m[1][2] = dir.y; r.m[1][3] = pos.y;
    r.m[2][0] = right.z; r.m[2][1] = new_up.z; r.m[2][2] = dir.z; r.m[2][3] = pos.z;
    r.m[3][0] = 0.0f;    r.m[3][1] = 0.0f;     r.m[3][2] = 0.0f;  r.m[3][3] = 1.0f;
    return r;
}

Transform to_bonsai(const Mat4 &m) {
    Transform t;
    t.r0 = float4{m.m[0][0], m.m[0][1], m.m[0][2], m.m[0][3]};
    t.r1 = float4{m.m[1][0], m.m[1][1], m.m[1][2], m.m[1][3]};
    t.r2 = float4{m.m[2][0], m.m[2][1], m.m[2][2], m.m[2][3]};
    t.r3 = float4{m.m[3][0], m.m[3][1], m.m[3][2], m.m[3][3]};
    return t;
}

} // namespace

int main(int argc, char **argv) {
    const char *output = (argc > 1) ? argv[1] : "pbrt_m0.ppm";

    int width = 400;
    int height = 225;
    if (const char *w = getenv("PBRT_WIDTH")) {
        width = atoi(w);
    }
    if (const char *h = getenv("PBRT_HEIGHT")) {
        height = atoi(h);
    }

    // pbrt: ProjectiveCamera's ctor. The screen window is the unit square in
    // the narrower dimension and the aspect ratio in the wider one.
    const float aspect = float(width) / float(height);
    float screen_min_x, screen_max_x, screen_min_y, screen_max_y;
    if (aspect > 1.0f) {
        screen_min_x = -aspect; screen_max_x = aspect;
        screen_min_y = -1.0f;   screen_max_y = 1.0f;
    } else {
        screen_min_x = -1.0f;          screen_max_x = 1.0f;
        screen_min_y = -1.0f / aspect; screen_max_y = 1.0f / aspect;
    }

    const float fov = 45.0f;
    const Mat4 screen_from_camera = perspective(fov, 1e-2f, 1000.0f);
    const Mat4 ndc_from_screen =
        scale(1.0f / (screen_max_x - screen_min_x),
              1.0f / (screen_min_y - screen_max_y), 1.0f) *
        translate(-screen_min_x, -screen_max_y, 0.0f);
    const Mat4 raster_from_ndc = scale(float(width), float(height), 1.0f);
    const Mat4 raster_from_screen = raster_from_ndc * ndc_from_screen;
    const Mat4 camera_from_raster =
        inverse(screen_from_camera) * inverse(raster_from_screen);

    // pbrt: the scene's camera-to-world transform. Looking down -z at a
    // sphere sitting at the origin.
    const Mat4 render_from_camera =
        look_at(Vec3{0.0f, 0.0f, 3.0f}, Vec3{0.0f, 0.0f, 0.0f},
                Vec3{0.0f, 1.0f, 0.0f});

    PerspectiveCamera camera;
    camera.camera_from_raster = to_bonsai(camera_from_raster);
    camera.render_from_camera = to_bonsai(render_from_camera);

    Sphere sphere;
    sphere.center = float3{0.0f, 0.0f, 0.0f};
    sphere.radius = 1.0f;

    const uint32_t npixels = uint32_t(width) * uint32_t(height);
    float3 *out = (float3 *)malloc(sizeof(float3) * npixels);

    render(camera, sphere, uint32_t(width), uint32_t(height), out);

    std::ofstream file(output);
    if (!file) {
        std::cerr << "cannot open " << output << " for writing\n";
        free(out);
        return 1;
    }
    file << "P3\n" << width << ' ' << height << "\n255\n";
    for (uint32_t p = 0; p < npixels; p++) {
        // pbrt: the film's write-out. No tone mapping and no gamma in M0, so
        // this is the clamp and the scale and nothing else.
        for (int c = 0; c < 3; c++) {
            const float v = out[p][c];
            const int b = int(256.0f * (v < 0.0f ? 0.0f : (v > 0.999f ? 0.999f : v)));
            file << b << (c == 2 ? '\n' : ' ');
        }
    }

    std::cout << "wrote " << output << " (" << width << 'x' << height << ")\n";
    free(out);
    return 0;
}
