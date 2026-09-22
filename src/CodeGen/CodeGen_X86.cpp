#include "CodeGen/CodeGen_X86.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/IntrinsicsX86.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#include <dlfcn.h>

#include <functional>

namespace bonsai {

llvm::Type *CodeGen_X86::indirect_return_type(const ir::Type &ret_type) {
    if (!ret_type.defined()) {
        return nullptr;
    }
    llvm::Type *type = codegen_type(ret_type);
    if (!type->isAggregateType()) {
        return nullptr;
    }
    // The leaves of the aggregate, as the return convention classes them: a
    // floating-point or vector leaf takes an SSE register, anything else an
    // integer one. x86-64's RetCC gives two of the first kind and three of
    // the second before it reaches for the x87 stack.
    struct Leaves {
        unsigned sse = 0;
        unsigned integer = 0;
    };
    const std::function<void(llvm::Type *, Leaves &)> count =
        [&](llvm::Type *t, Leaves &leaves) {
            if (auto *st = llvm::dyn_cast<llvm::StructType>(t)) {
                for (llvm::Type *element : st->elements()) {
                    count(element, leaves);
                }
            } else if (auto *at = llvm::dyn_cast<llvm::ArrayType>(t)) {
                for (uint64_t i = 0; i < at->getNumElements(); i++) {
                    count(at->getElementType(), leaves);
                }
            } else if (t->isFloatingPointTy() || t->isVectorTy()) {
                leaves.sse++;
            } else {
                leaves.integer++;
            }
        };
    Leaves leaves;
    count(type, leaves);
    if (leaves.sse <= 2 && leaves.integer <= 3) {
        return nullptr;
    }
    return type;
}

int CodeGen_X86::vector_register_bits() const {
    // The widest register the machine has: zmm with AVX-512, ymm with AVX,
    // xmm otherwise. A gang of sixteen 32-bit lanes is one zmm register, and
    // telling LLVM to prefer less (see optimize_module) would split every one
    // of its operations in two.
    if (has_feature("avx512f")) {
        return 512;
    }
    if (has_feature("avx")) {
        return 256;
    }
    return 128;
}

llvm::Value *CodeGen_X86::gather_indexed(llvm::Type *elem_llvm,
                                         llvm::Value *base,
                                         llvm::Value *indices, uint64_t scale,
                                         uint64_t disp, uint64_t align,
                                         uint32_t lanes, llvm::Value *mask,
                                         const std::string &name) {
    // Eight or sixteen 32-bit lanes with 32-bit indices at a scale the
    // addressing has, on a machine with the instruction: the shapes a gang's
    // gathers have. Eight lanes is a 256-bit gather, AVX-512VL's with its mask
    // in a k register or AVX2's with the mask in a vector; sixteen is the
    // 512-bit gather AVX-512F has. Anything else takes the generic form.
    auto *indices_t = llvm::dyn_cast<llvm::FixedVectorType>(indices->getType());
    const bool word = elem_llvm->isIntegerTy(32) || elem_llvm->isFloatTy();
    const bool scaled = scale == 1 || scale == 2 || scale == 4 || scale == 8;
    const bool avx512vl = has_feature("avx512vl");
    const bool eight = lanes == 8 && (avx512vl || has_feature("avx2"));
    const bool sixteen = lanes == 16 && has_feature("avx512f");
    if (!word || !scaled || !(eight || sixteen) || indices_t == nullptr ||
        indices_t->getNumElements() != lanes ||
        !indices_t->getElementType()->isIntegerTy(32)) {
        return CodeGen_LLVM::gather_indexed(elem_llvm, base, indices, scale, disp,
                                            align, lanes, mask, name);
    }

    // The displacement goes into the base: the intrinsic has base, index
    // and scale, and a scalar add is what a displacement costs.
    llvm::Value *at = base;
    if (disp != 0) {
        at = builder->CreateInBoundsGEP(
            i8_t, base, llvm::ConstantInt::get(i64_t, disp), name + "_base");
    }
    llvm::Type *vtype = llvm::FixedVectorType::get(elem_llvm, lanes);
    llvm::Value *on = mask != nullptr
                          ? mask
                          : llvm::Constant::getAllOnesValue(
                                llvm::FixedVectorType::get(i1_t, lanes));
    llvm::Value *off_lanes = llvm::Constant::getNullValue(vtype);
    const bool is_float = elem_llvm->isFloatTy();

    if (sixteen || avx512vl) {
        // vgatherdps/vpgatherdd {zmm,ymm}{k}, [base + index * scale]: the
        // mask is a k register, and the lanes it leaves off keep the first
        // operand, zero here as LLVM's passthrough is.
        const llvm::Intrinsic::ID id =
            sixteen ? (is_float ? llvm::Intrinsic::x86_avx512_mask_gather_dps_512
                                : llvm::Intrinsic::x86_avx512_mask_gather_dpi_512)
                    : (is_float ? llvm::Intrinsic::x86_avx512_mask_gather3siv8_sf
                                : llvm::Intrinsic::x86_avx512_mask_gather3siv8_si);
        return builder->CreateIntrinsic(
            id, {},
            {off_lanes, at, indices, on, llvm::ConstantInt::get(i32_t, scale)},
            nullptr, name);
    }
    // AVX2's gather takes its mask as a vector of the result's type, on in
    // the sign bit: the lanes to load are all ones.
    const llvm::Intrinsic::ID id = is_float
                                       ? llvm::Intrinsic::x86_avx2_gather_d_ps_256
                                       : llvm::Intrinsic::x86_avx2_gather_d_d_256;
    llvm::Value *sign =
        builder->CreateSExt(on, llvm::FixedVectorType::get(i32_t, lanes));
    if (is_float) {
        sign = builder->CreateBitCast(sign, vtype);
    }
    return builder->CreateIntrinsic(
        id, {}, {off_lanes, at, indices, sign, llvm::ConstantInt::get(i8_t, scale)},
        nullptr, name);
}

llvm::Value *CodeGen_X86::vector_int_division(llvm::Value *a, llvm::Value *b,
                                              bool is_signed, bool remainder) {
    // For lanes of 32 bits or fewer the quotient through doubles is exact:
    // for 0 <= a, b < 2^32 the correctly rounded double a/b is within
    // (a/b) 2^-53 of the truth, which is less than the 1/b that separates
    // a/b from the next integer on either side unless a/b is that integer, in
    // which case it is exact; so truncating it toward zero is the integer
    // quotient. A signed division is the same on magnitudes, which the
    // conversions carry the signs through. Narrower lanes fit a float by the
    // same argument. The remainder is a - qb. A zero divisor makes an
    // infinity the conversion has no value for, as the division it replaces
    // had none. A divisor known at compile time never gets here:
    // SSA/InvariantDivision.h has made it multiplies already.
    //
    // A division of scalars stays a `div`: one lane, one divide.
    auto *vt = llvm::dyn_cast<llvm::FixedVectorType>(a->getType());
    if (vt == nullptr || llvm::isa<llvm::Constant>(b)) {
        return nullptr;
    }
    const unsigned bits = vt->getScalarSizeInBits();
    if (bits > 32) {
        return nullptr;
    }
    llvm::Type *ft = llvm::FixedVectorType::get(bits > 16 ? f64_t : f32_t,
                                                vt->getNumElements());
    llvm::Value *fa = is_signed ? builder->CreateSIToFP(a, ft)
                                : builder->CreateUIToFP(a, ft);
    llvm::Value *fb = is_signed ? builder->CreateSIToFP(b, ft)
                                : builder->CreateUIToFP(b, ft);
    llvm::Value *fq = builder->CreateFDiv(fa, fb);
    llvm::Value *q = is_signed ? builder->CreateFPToSI(fq, vt, "quotient")
                               : builder->CreateFPToUI(fq, vt, "quotient");
    if (!remainder) {
        return q;
    }
    return builder->CreateSub(a, builder->CreateMul(q, b), "remainder");
}

namespace {

// The Vector Function ABI name glibc gives a vector entry point of libm:
// `_ZGV`, the ISA level (`b` SSE4, `c` AVX, `d` AVX2, `e` AVX-512), `N` for
// no mask, the lane count, a `v` per vector argument, and the scalar
// function's name.
std::string vector_abi_name(char isa, uint32_t lanes, unsigned arity,
                            const std::string &name, bool single) {
    return "_ZGV" + std::string(1, isa) + "N" + std::to_string(lanes) +
           std::string(arity, 'v') + "_" + name + (single ? "f" : "");
}

} // namespace

// Asked of the host's libmvec rather than assumed: which functions it has
// depends on the glibc -- 2.22 shipped sin, cos, exp, log and pow, 2.35 the
// rest of libm -- and which ISA levels run depends on the machine. Only a
// symbol present in this machine's libmvec, at a level this machine runs, is
// ever emitted, so nothing here can fail to link or to run where it was
// compiled; following the host means the code runs on the host.
void CodeGen_X86::probe_host_libraries() {
    if (!llvm::Triple(llvm::sys::getDefaultTargetTriple()).isOSLinux()) {
        return;
    }
    // Global, and left loaded: the JIT resolves what the code it runs calls
    // against this process's symbols, and libmvec is not otherwise among
    // them -- the compiler links libm, whose linker script pulls libmvec in
    // only for a program that references it.
    void *lib = dlopen("libmvec.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if (lib == nullptr) {
        return;
    }
    const llvm::StringMap<bool> features = llvm::sys::getHostCPUFeatures();
    const auto has = [&](const char *feature) {
        const auto it = features.find(feature);
        return it != features.end() && it->second;
    };
    struct Isa {
        char letter;
        uint32_t f32_lanes, f64_lanes;
        bool runs;
    };
    const Isa isas[] = {{'b', 4, 2, has("sse4.1")},
                        {'d', 8, 4, has("avx2")},
                        {'e', 16, 8, has("avx512f")}};
    struct Fn {
        const char *name;
        unsigned arity;
    };
    static const Fn fns[] = {
        {"sin", 1},   {"cos", 1},   {"tan", 1},   {"asin", 1},  {"acos", 1},
        {"atan", 1},  {"atan2", 2}, {"sinh", 1},  {"cosh", 1},  {"tanh", 1},
        {"asinh", 1}, {"acosh", 1}, {"atanh", 1}, {"exp", 1},   {"exp2", 1},
        {"exp10", 1}, {"expm1", 1}, {"log", 1},   {"log2", 1},  {"log10", 1},
        {"log1p", 1}, {"pow", 2},   {"hypot", 2}, {"cbrt", 1},  {"erf", 1},
        {"erfc", 1}};
    for (const Isa &isa : isas) {
        if (!isa.runs) {
            continue;
        }
        for (const Fn &fn : fns) {
            for (const bool single : {true, false}) {
                const std::string symbol =
                    vector_abi_name(isa.letter,
                                    single ? isa.f32_lanes : isa.f64_lanes,
                                    fn.arity, fn.name, single);
                if (dlsym(lib, symbol.c_str()) != nullptr) {
                    host_vector_math.insert(symbol);
                }
            }
        }
    }
}

std::optional<std::string>
CodeGen_X86::vector_math_symbol(const std::string &name, uint32_t lanes,
                                unsigned bits, unsigned arity) const {
    char isa = 0;
    const uint32_t f32_lanes = bits == 32 ? lanes : lanes * 2;
    switch (f32_lanes) {
    case 4:
        isa = 'b';
        break;
    case 8:
        isa = 'd';
        break;
    case 16:
        isa = 'e';
        break;
    default:
        return std::nullopt;
    }
    if (bits != 32 && bits != 64) {
        return std::nullopt;
    }
    const std::string symbol =
        vector_abi_name(isa, lanes, arity, name, bits == 32);
    if (host_vector_math.count(symbol) == 0) {
        return std::nullopt;
    }
    return symbol;
}

} // namespace bonsai
