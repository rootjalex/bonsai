/*
 * apps/wosx: the PDE WoSX's GPU solver is given, as WoSX asks for it -- a
 * module named wosx_pde_definition whose `PDEType` conforms to IPDE
 * (deps/wosx/include/wosx/gpu/core/pde.slang; the demo's
 * demo_apps/basic_2d/wosx-pde-definition.slang is the model). The Laplace
 * problem of wosx.bonsai: no source, no absorption, no reflecting boundary,
 * and the Dirichlet data `x^2 - y^2`, the harmonic function the driver's
 * CPU PDE and the program's `dirichlet` compute, so that the three agree.
 * Loaded at run time by WoSX's GPUTaskHandle from this directory, which the
 * driver names as its PDE directory.
 */

module wosx_pde_definition;
import wosx;

public struct HarmonicPDE : IPDE<float, 3>
{
    // The Laplace problem's Green's functions: harmonic, in three dimensions.
    public typedef HarmonicGreensFnFreeSpace3D GreensFnFreeSpace;
    public typedef HarmonicGreensFnBall3D GreensFnBall;

    // no source term
    public float source(float3 x)
    {
        return 0.0;
    }

    // the Dirichlet data: wosx.bonsai's `dirichlet`, wosx_hook.cpp's `dirichlet`
    public float dirichlet(float3 x, bool returnBoundaryNormalAlignedValue = false)
    {
        return x.x * x.x - x.y * x.y;
    }

    // no reflecting boundary
    public float robin(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {
        return 0.0;
    }

    public float robinCoeff(float3 x, float3 n, bool returnBoundaryNormalAlignedValue = false)
    {
        return 0.0;
    }

    // no absorption: Laplace, not screened Poisson
    public float absorptionCoeff()
    {
        return 0.0;
    }

    public bool isSourceConstant()
    {
        return true;
    }

    public bool areRobinConditionsPureNeumann()
    {
        return true;
    }

    public bool areRobinCoeffsNonnegative()
    {
        return true;
    }
};

// WoSX's GPU solvers take the module's PDE as `PDEType`.
public typedef HarmonicPDE PDEType;
