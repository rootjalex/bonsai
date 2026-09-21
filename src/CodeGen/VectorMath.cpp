#include "CodeGen/VectorMath.h"

#include "Utils.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/MDBuilder.h>

#include <cmath>
#include <limits>
#include <vector>

namespace bonsai {

namespace {

// How a lane's float is laid out.
struct Layout {
    unsigned bits;
    unsigned mantissa_bits;
    int bias;
};

Layout layout_of(llvm::Type *element) {
    if (element->isFloatTy()) {
        return {32, 23, 127};
    }
    if (element->isDoubleTy()) {
        return {64, 52, 1023};
    }
    internal_error << "vector math on a lane that is neither float nor double";
    return {};
}

llvm::Type *element_of(llvm::Type *type) {
    return llvm::cast<llvm::FixedVectorType>(type)->getElementType();
}

unsigned lanes_of(llvm::Type *type) {
    return llvm::cast<llvm::FixedVectorType>(type)->getNumElements();
}

// The coefficients below are written as the source writes them -- float
// literals for single precision -- and widened exactly.
std::vector<double> widen(std::initializer_list<float> coefficients) {
    return std::vector<double>(coefficients.begin(), coefficients.end());
}

constexpr double kPi = 3.141592653589793238462643383279502884;

} // namespace

bool VectorMath::handles(const std::string &name, llvm::Type *element) {
    if (element->isFloatTy()) {
        return name == "exp" || name == "log" || name == "sin" ||
               name == "cos" || name == "tan" || name == "asin" ||
               name == "acos" || name == "atan" || name == "atan2" ||
               name == "atanh" || name == "cosh" || name == "pow";
    }
    if (element->isDoubleTy()) {
        return name == "exp" || name == "log" || name == "sin" ||
               name == "cos" || name == "tan";
    }
    return false;
}

llvm::Value *VectorMath::call(const std::string &name,
                              llvm::ArrayRef<llvm::Value *> args) {
    internal_assert(!args.empty()) << "vector math " << name << " of nothing";
    for (llvm::Value *arg : args) {
        internal_assert(arg->getType() == args[0]->getType() &&
                        arg->getType()->isVectorTy())
            << "vector math " << name << " on arguments of differing shapes";
    }
    const bool unary = args.size() == 1;
    if (name == "exp" && unary) {
        return exp(args[0]);
    } else if (name == "log" && unary) {
        return log(args[0]);
    } else if (name == "sin" && unary) {
        return sin(args[0]);
    } else if (name == "cos" && unary) {
        return cos(args[0]);
    } else if (name == "tan" && unary) {
        return tan(args[0]);
    } else if (name == "asin" && unary) {
        return asin(args[0]);
    } else if (name == "acos" && unary) {
        return acos(args[0]);
    } else if (name == "atan" && unary) {
        return atan(args[0]);
    } else if (name == "atan2" && args.size() == 2) {
        return atan2(args[0], args[1]);
    } else if (name == "atanh" && unary) {
        return atanh(args[0]);
    } else if (name == "cosh" && unary) {
        return cosh(args[0]);
    } else if (name == "pow" && args.size() == 2) {
        return pow(args[0], args[1]);
    }
    internal_error << "vector math has no " << name << " of " << args.size()
                   << " arguments";
    return nullptr;
}

// --- The pieces -----------------------------------------------------------

llvm::Value *VectorMath::splat(llvm::Type *type, double c) {
    return llvm::ConstantFP::get(type, c);
}

llvm::Type *VectorMath::int_type_of(llvm::Type *type) {
    const Layout l = layout_of(element_of(type));
    return llvm::FixedVectorType::get(
        llvm::IntegerType::get(builder.getContext(), l.bits), lanes_of(type));
}

llvm::Value *VectorMath::fma(llvm::Value *a, llvm::Value *b, llvm::Value *c) {
    return builder.CreateIntrinsic(a->getType(), llvm::Intrinsic::fma,
                                   {a, b, c});
}

llvm::Value *VectorMath::unary(llvm::Intrinsic::ID id, llvm::Value *x) {
    return builder.CreateIntrinsic(x->getType(), id, {x});
}

llvm::Value *VectorMath::fabs(llvm::Value *x) {
    return unary(llvm::Intrinsic::fabs, x);
}

llvm::Value *VectorMath::is_nan(llvm::Value *x) {
    return builder.CreateFCmpUNO(x, x);
}

llvm::Value *VectorMath::is_inf(llvm::Value *x) {
    return builder.CreateFCmpOEQ(fabs(x),
                                 llvm::ConstantFP::getInfinity(x->getType()));
}

llvm::Value *VectorMath::sign_bit(llvm::Value *x) {
    llvm::Type *ints = int_type_of(x->getType());
    return builder.CreateICmpSLT(builder.CreateBitCast(x, ints),
                                 llvm::Constant::getNullValue(ints));
}

llvm::Value *VectorMath::polynomial(llvm::ArrayRef<double> coefficients,
                                    llvm::Value *x) {
    llvm::Type *t = x->getType();
    llvm::Value *y = splat(t, coefficients[0]);
    for (size_t i = 1; i < coefficients.size(); i++) {
        y = fma(x, y, splat(t, coefficients[i]));
    }
    return y;
}

llvm::Value *VectorMath::polynomial(llvm::ArrayRef<double> coefficients,
                                    llvm::Value *x, llvm::Value *x2) {
    llvm::Type *t = x->getType();
    const size_t n = coefficients.size();
    internal_assert(n >= 2) << "a two-chain polynomial needs two terms";
    llvm::Value *y0 = splat(t, coefficients[0]);
    llvm::Value *y1 = splat(t, coefficients[1]);
    for (size_t i = 2; i < n; i += 2) {
        y0 = fma(x2, y0, splat(t, coefficients[i]));
        if (i + 1 < n) {
            y1 = fma(x2, y1, splat(t, coefficients[i + 1]));
        }
    }
    return n % 2 == 1 ? fma(x, y1, y0) : fma(x, y0, y1);
}

llvm::Value *VectorMath::exp2_of_integer(llvm::Value *b) {
    // The float whose exponent field is b: b + bias, shifted up past the
    // mantissa. b is within [-bias, bias + 1] by the clamp before it; at the
    // top that is the infinity's pattern, at the bottom zero's.
    const Layout l = layout_of(element_of(b->getType()));
    llvm::Type *ints = int_type_of(b->getType());
    llvm::Value *e = builder.CreateFPToSI(b, ints);
    e = builder.CreateAdd(e, llvm::ConstantInt::get(ints, l.bias));
    e = builder.CreateShl(e, llvm::ConstantInt::get(ints, l.mantissa_bits));
    return builder.CreateBitCast(e, b->getType());
}

// --- exp --------------------------------------------------------------------

llvm::Value *VectorMath::exp(llvm::Value *x) {
    // XNNPACK's exp_taylor2 with m = 0: x = b ln 2 + r with b an integer and
    // |r| <= ln(2)/2, e^x = 2^b (1 + r + r^2 P(r)), P approximating the
    // series after two terms. ln 2 is split so that b ln 2 is subtracted
    // exactly enough; the clamp keeps 2^b representable, and the selects
    // put NaN and the overflow to infinity back.
    llvm::Type *t = x->getType();
    const Layout l = layout_of(element_of(t));
    const bool single = l.bits == 32;
    const std::vector<double> p =
        single ? widen({-4.9866619520e-05f, 2.2773783712e-04f,
                        1.4132461511e-03f, 8.3242934197e-03f,
                        4.1665628552e-02f, 1.6666725278e-01f,
                        4.9999994040e-01f})
               : std::vector<double>{
                     2.119890244143296032e-09, 2.521079371356939368e-08,
                     2.755585399809175769e-07, 2.755675838881680360e-06,
                     2.480158976814372031e-05, 1.984127071808468106e-04,
                     1.388888888680978647e-03, 8.333333332743943569e-03,
                     4.166666666668113195e-02, 1.666666666666796748e-01,
                     4.999999999999995559e-01};
    const double log2e = 1.4426950408889634;
    const double ln2_hi = single ? double(6.9314575195e-01f)
                                 : 6.93147180369123816490e-01;
    const double ln2_lo = single ? double(1.4286067653e-06f)
                                 : 1.90821492927058770002e-10;
    const double min_x = single ? -88.02969360351562 : -709.0895657128241;
    const double max_x = single ? 88.72283935546875 : 709.782712893384;

    llvm::Value *clamped = builder.CreateIntrinsic(
        t, llvm::Intrinsic::minnum,
        {builder.CreateIntrinsic(t, llvm::Intrinsic::maxnum,
                                 {x, splat(t, min_x)}),
         splat(t, max_x)});
    llvm::Value *b = unary(llvm::Intrinsic::roundeven,
                           builder.CreateFMul(clamped, splat(t, log2e)));
    llvm::Value *r = fma(b, splat(t, -ln2_hi), clamped);
    r = fma(b, splat(t, -ln2_lo), r);

    llvm::Value *exp2_b = exp2_of_integer(b);
    exp2_b = builder.CreateSelect(is_nan(x), x, exp2_b);

    llvm::Value *r2 = builder.CreateFMul(r, r);
    llvm::Value *exp2_r = fma(r2, polynomial(p, r, r2), r);
    llvm::Value *exp_x = fma(exp2_b, exp2_r, exp2_b);
    return builder.CreateSelect(is_inf(exp2_b), exp2_b, exp_x, "exp");
}

// --- log --------------------------------------------------------------------

llvm::Value *VectorMath::log_impl(llvm::Value *x, bool plus_one) {
    // XNNPACK's log_taylor2: y = 2^e m with m in [sqrt(2)/2, sqrt(2)) read
    // off the bits, log y = e ln 2 + log(1 + r) for r = m - 1, and
    // log(1 + r) = r + r^2 P(r)/Q(r). Added here: a denormal is scaled up
    // first, since the exponent field of one says nothing, and the special
    // values -- zero, a negative, an infinity, a NaN -- answered as libm
    // answers them.
    llvm::Type *t = x->getType();
    const Layout l = layout_of(element_of(t));
    const bool single = l.bits == 32;
    llvm::Type *ints = int_type_of(t);
    const std::vector<double> p =
        single ? widen({-5.2293352783e-03f, -3.3127889037e-01f,
                        -4.9999964237e-01f})
               : std::vector<double>{
                     -2.956046447212538203e-03, -6.691314126082094360e-02,
                     -4.289980732181625789e-01, -1.110965361416291097e+00,
                     -1.242991301338429500e+00, -4.999999999999990008e-01};
    const std::vector<double> q =
        single ? widen({3.9658042789e-01f, 1.3292170763e+00f, 1.0f})
               : std::vector<double>{
                     3.009181610003072455e-03, 7.942469932969810353e-02,
                     6.369060590899747742e-01, 2.230802780023532605e+00,
                     3.823696902394678077e+00, 3.152649269343555716e+00,
                     1.000000000000000000e+00};
    const double ln2 = 6.93147180559945309e-01;
    const double sqrt2 = 1.41421356237309504880;

    llvm::Value *one = splat(t, 1.0);
    llvm::Value *y = plus_one ? builder.CreateFAdd(x, one) : x;

    // A denormal, scaled into the normal range by 2^(mantissa bits + 1);
    // the exponent is adjusted back below.
    const double smallest_normal = std::ldexp(1.0, 1 - l.bias);
    const int scale_bits = int(l.mantissa_bits) + 1;
    llvm::Value *tiny = builder.CreateFCmpOLT(y, splat(t, smallest_normal));
    llvm::Value *scaled = builder.CreateSelect(
        tiny, builder.CreateFMul(y, splat(t, std::ldexp(1.0, scale_bits))), y);
    llvm::Value *adjust = builder.CreateSelect(tiny, splat(t, -scale_bits),
                                               splat(t, 0.0));

    llvm::Value *bits = builder.CreateBitCast(scaled, ints);
    llvm::Value *e = builder.CreateSIToFP(
        builder.CreateSub(
            builder.CreateAShr(bits,
                               llvm::ConstantInt::get(ints, l.mantissa_bits)),
            llvm::ConstantInt::get(ints, l.bias)),
        t);
    const uint64_t mantissa_mask = (uint64_t(1) << l.mantissa_bits) - 1;
    llvm::Value *m = builder.CreateBitCast(
        builder.CreateOr(
            builder.CreateAnd(bits, llvm::ConstantInt::get(ints, mantissa_mask)),
            builder.CreateBitCast(one, ints)),
        t);
    llvm::Value *large = builder.CreateFCmpOGT(m, splat(t, sqrt2));
    e = builder.CreateSelect(large, builder.CreateFAdd(e, one), e);
    m = builder.CreateSelect(large, builder.CreateFMul(m, splat(t, 0.5)), m);
    llvm::Value *r = builder.CreateFSub(m, one);
    if (plus_one) {
        // For y in [sqrt(2)/2, sqrt(2)) the residual is x itself, exactly,
        // where m - 1 would have lost its low bits to the addition.
        r = builder.CreateSelect(builder.CreateFCmpOEQ(e, splat(t, 0.0)), x,
                                 r);
    }
    llvm::Value *r2 = builder.CreateFMul(r, r);
    llvm::Value *log_r =
        fma(r2, builder.CreateFDiv(polynomial(p, r, r2), polynomial(q, r, r2)),
            r);
    llvm::Value *result =
        fma(builder.CreateFAdd(e, adjust), splat(t, ln2), log_r);

    // The special values, as libm has them: log 0 = -inf, log of a negative
    // or a NaN is NaN, log inf = inf.
    result = builder.CreateSelect(
        builder.CreateFCmpOEQ(y, splat(t, 0.0)),
        llvm::ConstantFP::getInfinity(t, /*Negative=*/true), result);
    result = builder.CreateSelect(builder.CreateFCmpULT(y, splat(t, 0.0)),
                                  llvm::ConstantFP::getNaN(t), result);
    result = builder.CreateSelect(
        builder.CreateFCmpOEQ(y, llvm::ConstantFP::getInfinity(t)),
        llvm::ConstantFP::getInfinity(t), result, plus_one ? "log1p" : "log");
    return result;
}

llvm::Value *VectorMath::log(llvm::Value *x) {
    return log_impl(x, /*plus_one=*/false);
}

llvm::Value *VectorMath::log1p(llvm::Value *x) {
    return log_impl(x, /*plus_one=*/true);
}

// --- sin, cos, tan ------------------------------------------------------------

VectorMath::SinCos VectorMath::sincos(llvm::Value *x) {
    // XNNPACK's range_reduce and sin_impl: x = k pi/2 + x' with |x'| <= pi/4,
    // pi/2 held in three parts so that k pi/2 comes off exactly enough; then
    // by quadrant, sin x is sin x', sin(pi/2 - |x'|) mirrored, or either
    // negated, and cos x is sin of the next quadrant. One odd polynomial in
    // x'^2 serves both.
    llvm::Type *t = x->getType();
    const Layout l = layout_of(element_of(t));
    const bool single = l.bits == 32;
    llvm::Type *ints = int_type_of(t);
    const std::vector<double> half_pi =
        single ? widen({-1.57079637e+00f, 4.37113883e-08f, 1.71512451e-15f})
               : std::vector<double>{-1.5707963267948966e+00,
                                     -6.1232339957367660e-17,
                                     1.4973849048591698e-33};
    const double max_x = single ? 1e6 : 1e12;
    const std::vector<double> p =
        single ? widen({2.6052248359e-06f, -1.9809075457e-04f,
                        8.3330506459e-03f, -1.6666658223e-01f, 1.0f})
               : std::vector<double>{
                     2.765834900803161893e-15, -7.650022465844408508e-13,
                     1.605934858160708274e-10, -2.505211878738542593e-08,
                     2.755731940097233417e-06, -1.984126984283516356e-04,
                     8.333333333339692714e-03, -1.666666666666674068e-01,
                     1.000000000000000000e+00};

    // The lanes the reduction covers; a NaN is not one, and neither is an
    // infinity, and both come out NaN below without a rare path: the
    // integer k of a lane out of range is taken as zero, so the argument
    // itself flows through the polynomial.
    llvm::Value *magnitude = fabs(x);
    llvm::Value *in_range = builder.CreateFCmpOLE(magnitude, splat(t, max_x));
    llvm::Value *rare = builder.CreateAnd(
        builder.CreateFCmpOGT(magnitude, splat(t, max_x)),
        builder.CreateNot(is_inf(x)), "sincos_rare");

    llvm::Value *kf = unary(llvm::Intrinsic::roundeven,
                            builder.CreateFMul(x, splat(t, 2.0 / kPi)));
    kf = builder.CreateSelect(in_range, kf, splat(t, 0.0));
    llvm::Value *k = builder.CreateFPToSI(kf, ints);
    llvm::Value *xr = x;
    for (const double part : half_pi) {
        xr = fma(kf, splat(t, part), xr);
    }
    llvm::Value *mirror = builder.CreateFSub(splat(t, kPi / 2), fabs(xr));
    llvm::Value *zero = llvm::Constant::getNullValue(ints);
    llvm::Value *one = llvm::ConstantInt::get(ints, 1);
    llvm::Value *two = llvm::ConstantInt::get(ints, 2);
    const auto by_quadrant = [&](llvm::Value *quadrant) {
        llvm::Value *v = builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateAnd(quadrant, one), zero), xr,
            mirror);
        return builder.CreateSelect(
            builder.CreateICmpEQ(builder.CreateAnd(quadrant, two), zero), v,
            builder.CreateFNeg(v));
    };
    llvm::Value *xs = by_quadrant(k);
    llvm::Value *xc = by_quadrant(builder.CreateAdd(k, one));
    const auto odd_polynomial = [&](llvm::Value *v) {
        return builder.CreateFMul(polynomial(p, builder.CreateFMul(v, v)), v);
    };
    llvm::Value *s = odd_polynomial(xs);
    llvm::Value *c = odd_polynomial(xc);
    llvm::Value *nan = llvm::ConstantFP::getNaN(t);
    llvm::Value *infinite = is_inf(x);
    s = builder.CreateSelect(infinite, nan, s);
    c = builder.CreateSelect(infinite, nan, c);
    return {s, c, rare};
}

llvm::Value *VectorMath::finish_rare(llvm::Value *fast, llvm::Value *rare,
                                     llvm::Value *x, const std::string &name) {
    // The rare lanes -- finite, past the range reduction -- taken a lane at
    // a time by the scalar function, in a block the gang enters only if
    // some lane is rare. Every lane is computed there and the rare ones
    // selected, so that a lane's answer never depends on what another lane
    // held.
    llvm::LLVMContext &ctx = builder.getContext();
    llvm::Function *function = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *from = builder.GetInsertBlock();
    llvm::BasicBlock *slow =
        llvm::BasicBlock::Create(ctx, name + "_rare", function);
    llvm::BasicBlock *join =
        llvm::BasicBlock::Create(ctx, name + "_join", function);
    llvm::MDBuilder md(ctx);
    builder.CreateCondBr(builder.CreateOrReduce(rare), slow, join,
                         md.createBranchWeights(1, 1 << 20));

    builder.SetInsertPoint(slow);
    llvm::Value *all = fast;
    for (unsigned lane = 0; lane < lanes_of(x->getType()); lane++) {
        llvm::Value *v = builder.CreateExtractElement(x, uint64_t(lane));
        all = builder.CreateInsertElement(all, scalar(name, {v}),
                                          uint64_t(lane));
    }
    llvm::Value *fixed = builder.CreateSelect(rare, all, fast);
    builder.CreateBr(join);

    builder.SetInsertPoint(join);
    llvm::PHINode *phi = builder.CreatePHI(fast->getType(), 2, name);
    phi->addIncoming(fast, from);
    phi->addIncoming(fixed, slow);
    return phi;
}

llvm::Value *VectorMath::sin(llvm::Value *x) {
    const SinCos sc = sincos(x);
    return finish_rare(sc.sin, sc.rare, x, "sin");
}

llvm::Value *VectorMath::cos(llvm::Value *x) {
    const SinCos sc = sincos(x);
    return finish_rare(sc.cos, sc.rare, x, "cos");
}

llvm::Value *VectorMath::tan(llvm::Value *x) {
    const SinCos sc = sincos(x);
    return finish_rare(builder.CreateFDiv(sc.sin, sc.cos), sc.rare, x, "tan");
}

// --- asin, acos, atan, atan2 ----------------------------------------------------

llvm::Value *VectorMath::asin_core(llvm::Value *z, llvm::Value *w) {
    // Cephes asinf: asin w = w + w z P(z) for z = w^2 up to a quarter.
    const std::vector<double> p =
        widen({4.2163199048E-2f, 2.4181311049E-2f, 4.5470025998E-2f,
               7.4953002686E-2f, 1.6666752422E-1f});
    return fma(builder.CreateFMul(polynomial(p, z), z), w, w);
}

llvm::Value *VectorMath::asin(llvm::Value *x) {
    // Below 0.5 the polynomial on x itself; above, asin |x| = pi/2 -
    // 2 asin(sqrt((1 - |x|)/2)), the half-angle, whose argument is small.
    // An argument past one makes the square root NaN, as libm answers it.
    llvm::Type *t = x->getType();
    llvm::Value *a = fabs(x);
    llvm::Value *big = builder.CreateFCmpOGT(a, splat(t, 0.5));
    llvm::Value *z = builder.CreateSelect(
        big, fma(a, splat(t, -0.5), splat(t, 0.5)), builder.CreateFMul(a, a));
    llvm::Value *w =
        builder.CreateSelect(big, unary(llvm::Intrinsic::sqrt, z), a);
    llvm::Value *y = asin_core(z, w);
    y = builder.CreateSelect(
        big, fma(y, splat(t, -2.0), splat(t, kPi / 2)), y);
    return builder.CreateIntrinsic(t, llvm::Intrinsic::copysign, {y, x},
                                   nullptr, "asin");
}

llvm::Value *VectorMath::acos(llvm::Value *x) {
    // Cephes acosf: 2 asin(sqrt((1 - x)/2)) above 0.5, pi minus that below
    // -0.5, and pi/2 - asin x between, all from the one polynomial.
    llvm::Type *t = x->getType();
    llvm::Value *a = fabs(x);
    llvm::Value *big = builder.CreateFCmpOGT(a, splat(t, 0.5));
    llvm::Value *z = builder.CreateSelect(
        big, fma(a, splat(t, -0.5), splat(t, 0.5)), builder.CreateFMul(a, a));
    llvm::Value *w =
        builder.CreateSelect(big, unary(llvm::Intrinsic::sqrt, z), a);
    llvm::Value *y = asin_core(z, w);
    llvm::Value *twice = builder.CreateFAdd(y, y);
    llvm::Value *outer = builder.CreateSelect(
        builder.CreateFCmpOGT(x, splat(t, 0.0)), twice,
        builder.CreateFSub(splat(t, kPi), twice));
    llvm::Value *inner = builder.CreateFSub(
        splat(t, kPi / 2),
        builder.CreateIntrinsic(t, llvm::Intrinsic::copysign, {y, x}));
    return builder.CreateSelect(big, outer, inner, "acos");
}

llvm::Value *VectorMath::atan_of_magnitude(llvm::Value *u) {
    // Cephes atanf: fold u past tan(3pi/8) to -1/u plus pi/2, past tan(pi/8)
    // to (u - 1)/(u + 1) plus pi/4, then v + v z P(z) for z = v^2. One
    // division, of a numerator and a denominator chosen by the fold.
    llvm::Type *t = u->getType();
    const std::vector<double> p =
        widen({8.05374449538e-2f, -1.38776856032e-1f, 1.99777106478e-1f,
               -3.33329491539e-1f});
    llvm::Value *one = splat(t, 1.0);
    llvm::Value *big = builder.CreateFCmpOGT(u, splat(t, 2.414213562373095));
    llvm::Value *mid = builder.CreateFCmpOGT(u, splat(t, 0.4142135623730950));
    llvm::Value *numerator = builder.CreateSelect(
        big, splat(t, -1.0),
        builder.CreateSelect(mid, builder.CreateFSub(u, one), u));
    llvm::Value *denominator = builder.CreateSelect(
        big, u, builder.CreateSelect(mid, builder.CreateFAdd(u, one), one));
    llvm::Value *v = builder.CreateFDiv(numerator, denominator);
    llvm::Value *z = builder.CreateFMul(v, v);
    llvm::Value *r = fma(builder.CreateFMul(polynomial(p, z), z), v, v);
    llvm::Value *offset = builder.CreateSelect(
        big, splat(t, kPi / 2),
        builder.CreateSelect(mid, splat(t, kPi / 4), splat(t, 0.0)));
    return builder.CreateFAdd(r, offset);
}

llvm::Value *VectorMath::atan(llvm::Value *x) {
    return builder.CreateIntrinsic(x->getType(), llvm::Intrinsic::copysign,
                                   {atan_of_magnitude(fabs(x)), x}, nullptr,
                                   "atan");
}

llvm::Value *VectorMath::atan2(llvm::Value *y, llvm::Value *x) {
    // The angle of (x, y): atan(|y|/|x|) in the first quadrant, reflected
    // to the second when x is negative -- by its sign bit, so that -0 counts
    // -- and given y's sign. The quotient answers the axes on its own: an x
    // of zero makes it infinite and the angle pi/2, a y of zero makes it
    // zero. Both zero, or both infinite, is a NaN quotient and is answered
    // as C does: 0 or pi by x's sign, and pi/4 from the axis.
    llvm::Type *t = x->getType();
    llvm::Value *ay = fabs(y);
    llvm::Value *ax = fabs(x);
    llvm::Value *zero = splat(t, 0.0);
    llvm::Value *angle = atan_of_magnitude(builder.CreateFDiv(ay, ax));
    llvm::Value *both_zero = builder.CreateAnd(builder.CreateFCmpOEQ(ay, zero),
                                               builder.CreateFCmpOEQ(ax, zero));
    llvm::Value *both_infinite =
        builder.CreateAnd(is_inf(y), is_inf(x));
    angle = builder.CreateSelect(both_zero, zero, angle);
    angle = builder.CreateSelect(both_infinite, splat(t, kPi / 4), angle);
    angle = builder.CreateSelect(sign_bit(x),
                                 builder.CreateFSub(splat(t, kPi), angle), angle);
    return builder.CreateIntrinsic(t, llvm::Intrinsic::copysign, {angle, y},
                                   nullptr, "atan2");
}

// --- atanh, cosh, pow -----------------------------------------------------------

llvm::Value *VectorMath::atanh(llvm::Value *x) {
    // Cephes atanhf below 0.5: x + x^3 P(x^2). Beyond, atanh x = log1p(2x /
    // (1 - x)) / 2, log1p keeping the precision near the pole, where 1 - x
    // is small; at one the quotient is infinite and so is the answer, past
    // one it is negative below -1 and the logarithm NaN, as libm has both.
    llvm::Type *t = x->getType();
    const std::vector<double> p =
        widen({1.81740078349E-1f, 8.24370301058E-2f, 1.46691431730E-1f,
               1.99782164500E-1f, 3.33337300303E-1f});
    llvm::Value *a = fabs(x);
    llvm::Value *small = builder.CreateFCmpOLT(a, splat(t, 0.5));
    llvm::Value *z = builder.CreateFMul(x, x);
    llvm::Value *near = fma(builder.CreateFMul(polynomial(p, z), z), x, x);
    llvm::Value *quotient = builder.CreateFDiv(
        builder.CreateFAdd(a, a), builder.CreateFSub(splat(t, 1.0), a));
    llvm::Value *far = builder.CreateIntrinsic(
        t, llvm::Intrinsic::copysign,
        {builder.CreateFMul(log1p(quotient), splat(t, 0.5)), x});
    return builder.CreateSelect(small, near, far, "atanh");
}

llvm::Value *VectorMath::cosh(llvm::Value *x) {
    // (e^|x| + e^-|x|) / 2, Cephes coshf's form: past 89 the exponential
    // is infinite and so is the answer.
    llvm::Type *t = x->getType();
    llvm::Value *e = exp(fabs(x));
    llvm::Value *result =
        fma(splat(t, 0.5), e, builder.CreateFDiv(splat(t, 0.5), e));
    result->setName("cosh");
    return result;
}

llvm::Value *VectorMath::pow(llvm::Value *x, llvm::Value *y) {
    // exp(y log |x|) in double, so that the error of the logarithm, which
    // y multiplies, stays below a single float's last place. Then C's
    // conventions: a negative base to an odd integer power is negative, to a
    // non-integer power NaN; anything to the zeroth power is one, and one to
    // any power is one, NaN included.
    llvm::Type *t = x->getType();
    llvm::Type *wide = llvm::FixedVectorType::get(
        llvm::Type::getDoubleTy(builder.getContext()), lanes_of(t));
    const bool widen_it = element_of(t)->isFloatTy();
    llvm::Value *xd = widen_it ? builder.CreateFPExt(x, wide) : x;
    llvm::Value *yd = widen_it ? builder.CreateFPExt(y, wide) : y;
    llvm::Type *d = xd->getType();
    llvm::Value *zero = splat(d, 0.0);
    llvm::Value *one = splat(d, 1.0);

    llvm::Value *r = exp(builder.CreateFMul(yd, log(fabs(xd))));

    llvm::Value *floor_y = unary(llvm::Intrinsic::floor, yd);
    llvm::Value *y_integer = builder.CreateFCmpOEQ(floor_y, yd);
    llvm::Value *half_y = builder.CreateFMul(yd, splat(d, 0.5));
    llvm::Value *y_odd = builder.CreateAnd(
        y_integer, builder.CreateFCmpONE(
                       unary(llvm::Intrinsic::floor, half_y), half_y));
    llvm::Value *negative = sign_bit(xd);
    r = builder.CreateSelect(builder.CreateAnd(negative, y_odd),
                             builder.CreateFNeg(r), r);
    r = builder.CreateSelect(
        builder.CreateAnd(
            builder.CreateAnd(negative, builder.CreateFCmpONE(xd, zero)),
            builder.CreateNot(y_integer)),
        llvm::ConstantFP::getNaN(d), r);
    r = builder.CreateSelect(builder.CreateFCmpOEQ(yd, zero), one, r);
    r = builder.CreateSelect(builder.CreateFCmpOEQ(xd, one), one, r);
    return widen_it ? builder.CreateFPTrunc(r, t, "pow") : r;
}

} // namespace bonsai
