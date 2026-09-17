#include "Lower/WordStorage.h"

#include "Error.h"
#include "IR/Equality.h"
#include "IR/Printer.h"

#include <algorithm>

namespace bonsai {
namespace lower {

namespace {

using namespace ir;

Type u32() { return UInt_t::make(32); }
Type u64() { return UInt_t::make(64); }

Expr u32_imm(uint64_t v) { return UIntImm::make(u32(), v); }

// Word `k` of `words`.
Expr word(const Expr &words, uint64_t k) {
    return Extract::make(words, u32_imm(k));
}

// The unsigned integer with the same bits as `type`, for reinterpreting a
// value of it into words and back.
Type bits_of(const Type &type) {
    if (type.is_float()) {
        return UInt_t::make(type.as<Float_t>()->bits());
    }
    if (type.is<Int_t>()) {
        return UInt_t::make(type.as<Int_t>()->bits);
    }
    internal_assert(type.is<UInt_t>()) << "no bits for " << type;
    return type;
}

// `value` as the unsigned integer of its bits.
Expr as_bits(const Expr &value, const Type &type) {
    const Type bits = bits_of(type);
    return bits.same_as(type) || equals(bits, type)
               ? value
               : Cast::make(bits, value, Cast::Mode::Reinterpret);
}

// The unsigned integer `bits` reinterpreted as `type`.
Expr from_bits(const Expr &bits, const Type &type) {
    return equals(bits.type(), type)
               ? bits
               : Cast::make(type, bits, Cast::Mode::Reinterpret);
}

bool is_zero_word(const Expr &e) {
    const UIntImm *imm = e.as<UIntImm>();
    return imm != nullptr && imm->value == 0;
}

// ORs `piece` (a u32) into word `k`: replaces a word still zero, ORs into
// one that already holds another field's bits.
void or_into(std::vector<Expr> &words, uint64_t k, Expr piece) {
    internal_assert(k < words.size())
        << "word " << k << " of " << words.size();
    words[k] = is_zero_word(words[k])
                   ? std::move(piece)
                   : BinOp::make(BinOp::OpType::BwOr, words[k], std::move(piece));
}

} // namespace

uint64_t words_of(const Type &type) { return (layout_bytes(type) + 3) / 4; }

Type words_type(uint64_t n) {
    return Vector_t::make(u32(), uint32_t(n), /*packed=*/true);
}

Expr read_words(const Expr &words, uint64_t offset, const Type &type) {
    if (const Struct_t *s = type.as<Struct_t>()) {
        std::vector<Expr> fields;
        fields.reserve(s->fields.size());
        for (size_t i = 0; i < s->fields.size(); i++) {
            internal_assert(!s->fields[i].type.is<Ref_t>())
                << "[unimplemented] a variant's field " << s->fields[i].name
                << " is a reference, which has no storage to read";
            fields.push_back(read_words(words, offset + layout_offset(*s, i),
                                        s->fields[i].type));
        }
        return Build::make(type, std::move(fields));
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        // Components back to back, whether the vector is packed storage or
        // a value that occupies more than its components (see
        // Type::bytes); either way component i is at i times its size.
        const uint64_t step = layout_bytes(v->etype);
        std::vector<Expr> components;
        components.reserve(v->lanes);
        for (uint32_t i = 0; i < v->lanes; i++) {
            components.push_back(read_words(words, offset + i * step, v->etype));
        }
        return Build::make(type, std::move(components));
    }
    const uint64_t k = offset / 4;
    const uint64_t within = offset % 4;
    if (type.is<Bool_t>()) {
        // A byte, zero or one.
        Expr piece = BinOp::make(
            BinOp::OpType::BwAnd,
            BinOp::make(BinOp::OpType::Shr, word(words, k), u32_imm(within * 8)),
            u32_imm(0xff));
        return BinOp::make(BinOp::OpType::Neq, std::move(piece), u32_imm(0));
    }
    internal_assert(type.is_int_or_uint() || type.is_float())
        << "[unimplemented] reading a " << type << " out of words";
    const uint64_t size = layout_bytes(type);
    if (size == 4) {
        internal_assert(within == 0)
            << "a four-byte field at byte " << offset << ", off a word";
        return from_bits(word(words, k), type);
    }
    if (size == 8) {
        internal_assert(within == 0)
            << "an eight-byte field at byte " << offset << ", off a word";
        Expr lo = Cast::make(u64(), word(words, k));
        Expr hi = Cast::make(u64(), word(words, k + 1));
        Expr bits = BinOp::make(
            BinOp::OpType::BwOr, std::move(lo),
            BinOp::make(BinOp::OpType::Shl, std::move(hi),
                        UIntImm::make(u64(), 32)));
        return from_bits(std::move(bits), type);
    }
    internal_assert(size == 1 || size == 2)
        << "[unimplemented] a " << size << "-byte field in words";
    internal_assert(within + size <= 4)
        << "a " << size << "-byte field at byte " << offset
        << " straddles a word";
    Expr piece = BinOp::make(
        BinOp::OpType::BwAnd,
        BinOp::make(BinOp::OpType::Shr, word(words, k), u32_imm(within * 8)),
        u32_imm((1ull << (size * 8)) - 1));
    // Down to the field's width, then to its type: a truncation, then a
    // reinterpret for a signed field.
    Expr narrow = Cast::make(bits_of(type), std::move(piece));
    return from_bits(std::move(narrow), type);
}

void write_words(std::vector<Expr> &words, uint64_t offset, const Expr &value,
                 const Type &type) {
    // A value built here from its parts is taken apart into them, rather
    // than read back field by field out of a build.
    const Build *built = value.as<Build>();
    if (const Struct_t *s = type.as<Struct_t>()) {
        const bool whole = built != nullptr &&
                           built->values.size() == s->fields.size();
        for (size_t i = 0; i < s->fields.size(); i++) {
            internal_assert(!s->fields[i].type.is<Ref_t>())
                << "[unimplemented] a variant's field " << s->fields[i].name
                << " is a reference, which has no storage to write";
            write_words(words, offset + layout_offset(*s, i),
                        whole ? built->values[i]
                              : Access::make(s->fields[i].name, value),
                        s->fields[i].type);
        }
        return;
    }
    if (const Vector_t *v = type.as<Vector_t>()) {
        const bool whole = built != nullptr && built->values.size() == v->lanes;
        const uint64_t step = layout_bytes(v->etype);
        for (uint32_t i = 0; i < v->lanes; i++) {
            write_words(words, offset + i * step,
                        whole ? built->values[i] : Extract::make(value, u32_imm(i)),
                        v->etype);
        }
        return;
    }
    const uint64_t k = offset / 4;
    const uint64_t within = offset % 4;
    if (type.is<Bool_t>()) {
        Expr byte = Select::make(value, u32_imm(1), u32_imm(0));
        or_into(words, k,
                within == 0 ? std::move(byte)
                            : BinOp::make(BinOp::OpType::Shl, std::move(byte),
                                          u32_imm(within * 8)));
        return;
    }
    internal_assert(type.is_int_or_uint() || type.is_float())
        << "[unimplemented] writing a " << type << " into words";
    const uint64_t size = layout_bytes(type);
    if (size == 4) {
        internal_assert(within == 0)
            << "a four-byte field at byte " << offset << ", off a word";
        or_into(words, k, as_bits(value, type));
        return;
    }
    if (size == 8) {
        internal_assert(within == 0)
            << "an eight-byte field at byte " << offset << ", off a word";
        Expr bits = as_bits(value, type);
        or_into(words, k, Cast::make(u32(), bits));
        or_into(words, k + 1,
                Cast::make(u32(), BinOp::make(BinOp::OpType::Shr, bits,
                                              UIntImm::make(u64(), 32))));
        return;
    }
    internal_assert(size == 1 || size == 2)
        << "[unimplemented] a " << size << "-byte field in words";
    internal_assert(within + size <= 4)
        << "a " << size << "-byte field at byte " << offset
        << " straddles a word";
    // Up to a word: the field's bits, zero-extended, shifted to its place.
    Expr piece = Cast::make(u32(), as_bits(value, type));
    or_into(words, k,
            within == 0 ? std::move(piece)
                        : BinOp::make(BinOp::OpType::Shl, std::move(piece),
                                      u32_imm(within * 8)));
}

} // namespace lower
} // namespace bonsai
