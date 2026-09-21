#pragma once

#include <llvm/IR/IRBuilder.h>

#include <functional>
#include <string>

namespace bonsai {

// libm's transcendental functions on a vector of lanes, built inline as LLVM
// IR: a range reduction, a polynomial or a rational approximation, and the
// selects that put the special values back, with no branch that depends on a
// lane's value. The same IR lowers on every target -- x86, and a GPU, whose
// device library is scalar -- and it is what a gang's `sin` becomes instead
// of a call.
//
// Why not the machine's vector math library. glibc's libmvec has vector entry
// points for all of these, and a gang called them through LLVM 22. They are
// fast on ordinary arguments and answer a special one -- a NaN, an infinity,
// a cosine outside [-1, 1] -- by handing every such lane to the scalar
// function one at a time, through glibc's error machinery, and in a gang some
// lanes are always off and hold whatever they last held: in a packet render
// half the lanes reaching acosf were off lanes carrying a value outside its
// domain, and one in twenty reaching sinf. The linearizer replaces an off
// lane's argument with a benign one (see SSA/Linearize.cpp), but LLVM treats
// its `llvm.sin` as a pure function and moves that select through it, so
// the library saw the lanes anyway. Code with no data-dependent branch
// answers every lane at the same speed, which is what a gang needs; it also
// owes nothing to the glibc version, and a GPU has no libmvec at all.
//
// Where the arithmetic comes from.
//
//   * exp, log, sin, cos and tan are XNNPACK's, from ynnpack's target
//     independent SIMD library (google/XNNPACK, ynnpack/base/simd/
//     target_independent.inc, BSD licensed): exp as 2^k times a degree-seven
//     polynomial in the residual, with ln 2 split in two for the reduction;
//     log as the exponent times ln 2 plus a rational approximation on the
//     mantissa scaled into [sqrt(2)/2, sqrt(2)); sin and cos by reducing to
//     [-pi/4, pi/4] against pi/2 split in three, mirroring by quadrant, and
//     one odd polynomial for both. In single and in double precision, the
//     coefficients theirs.
//   * asin, acos, atan and atan2 follow Cephes (S. L. Moshier, Cephes Math
//     Library, single-precision asinf.c, atanf.c and atan2f.c; the same
//     coefficients ispc's and Eigen's take): asin by a polynomial in x^2 up
//     to 0.5 and the half-angle sqrt((1 - |x|)/2) beyond it, acos from the
//     same polynomial, atan by folding the argument at tan(pi/8) and
//     tan(3pi/8) and a polynomial in the remainder, atan2 from atan by
//     quadrant.
//   * atanh is Cephes's polynomial below 0.5 and log1p(2x/(1 - x))/2 beyond
//     it, log1p being XNNPACK's; cosh is (e^|x| + e^-|x|)/2 from exp; pow is
//     exp(y log x) taken in double precision, with the sign of a negative
//     base to an odd power put back and the identities pow(x, 0) = 1 and
//     pow(1, y) = 1 kept.
//
// Accuracy, measured against glibc's libm (see tests/bonsai/correctness/
// llvm/vector-math.bonsai): within a few units in the last place everywhere
// in the domain, which is the accuracy libmvec promised. Not bit for bit
// with libm -- no vector library is, and apps/pbrt already compared its
// gangs against pbrt on libmvec's answers.
//
// Beyond the range reduction. sin and cos reduce arguments up to 1e6 (1e12
// in double) as XNNPACK does; a finite argument past that -- which a
// renderer never produces -- is taken a lane at a time by the scalar function
// the caller supplies, libm's on the CPU, behind a branch the whole gang
// takes only if some lane needs it. An infinite argument is a NaN inline, as
// libm answers it. Nothing else here has a rare path.
//
// Double precision has what the source provides: exp, log, sin, cos and
// tan. A vector of doubles asking for the rest goes to libm as before.
struct VectorMath {
    // The function `name` of libm -- "sin", "atan2" -- applied to scalar
    // arguments, for the lanes a range reduction does not cover: the caller
    // knows what a scalar call is on its target.
    using ScalarCall = std::function<llvm::Value *(
        const std::string &name, llvm::ArrayRef<llvm::Value *> args)>;

    VectorMath(llvm::IRBuilder<> &builder, ScalarCall scalar)
        : builder(builder), scalar(std::move(scalar)) {}

    // Is `name` built here for a vector of `element` lanes?
    static bool handles(const std::string &name, llvm::Type *element);

    // libm's `name` on `args`, vectors of one type: the result has that type.
    llvm::Value *call(const std::string &name,
                      llvm::ArrayRef<llvm::Value *> args);

    llvm::Value *exp(llvm::Value *x);
    llvm::Value *log(llvm::Value *x);
    llvm::Value *log1p(llvm::Value *x);
    llvm::Value *sin(llvm::Value *x);
    llvm::Value *cos(llvm::Value *x);
    llvm::Value *tan(llvm::Value *x);
    llvm::Value *asin(llvm::Value *x);
    llvm::Value *acos(llvm::Value *x);
    llvm::Value *atan(llvm::Value *x);
    llvm::Value *atan2(llvm::Value *y, llvm::Value *x);
    llvm::Value *atanh(llvm::Value *x);
    llvm::Value *cosh(llvm::Value *x);
    llvm::Value *pow(llvm::Value *x, llvm::Value *y);

  private:
    llvm::IRBuilder<> &builder;
    ScalarCall scalar;

    // A constant `c` in every lane of `type`.
    llvm::Value *splat(llvm::Type *type, double c);
    // The vector of integers as wide as `type`'s lanes.
    llvm::Type *int_type_of(llvm::Type *type);
    llvm::Value *fma(llvm::Value *a, llvm::Value *b, llvm::Value *c);
    llvm::Value *unary(llvm::Intrinsic::ID id, llvm::Value *x);
    llvm::Value *fabs(llvm::Value *x);
    llvm::Value *is_nan(llvm::Value *x);
    llvm::Value *is_inf(llvm::Value *x);
    llvm::Value *sign_bit(llvm::Value *x);
    // Horner's rule over `coefficients`, highest power first.
    llvm::Value *polynomial(llvm::ArrayRef<double> coefficients,
                            llvm::Value *x);
    // The same polynomial evaluated as two chains in x^2 joined by x, for
    // more instruction-level parallelism (XNNPACK's eval_polynomial).
    llvm::Value *polynomial(llvm::ArrayRef<double> coefficients, llvm::Value *x,
                            llvm::Value *x2);
    // 2^b for an integer-valued float b within the exponent's range.
    llvm::Value *exp2_of_integer(llvm::Value *b);
    llvm::Value *log_impl(llvm::Value *x, bool plus_one);
    // {sin x, cos x}, and the lanes whose argument the reduction did not
    // cover, which the caller finishes with `scalar`.
    struct SinCos {
        llvm::Value *sin, *cos, *rare;
    };
    SinCos sincos(llvm::Value *x);
    // `fast` with the lanes `rare` selects recomputed by `name`'s scalar
    // function, behind a branch taken only if some lane is rare.
    llvm::Value *finish_rare(llvm::Value *fast, llvm::Value *rare,
                             llvm::Value *x, const std::string &name);
    // asin(w) for w in [0, ~0.71] as w + w z P(z), z = w^2, given z and w.
    llvm::Value *asin_core(llvm::Value *z, llvm::Value *w);
    // atan(u) for u >= 0.
    llvm::Value *atan_of_magnitude(llvm::Value *u);
};

} // namespace bonsai
