#pragma once

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace bonsai {
namespace ir {

// What a piece of lowered code was in the program as written, so that a
// schedule can point at the construct and lowering can find the code.
//
// The arm of a `match` is written once, in one function. Lower/ADTs.cpp turns
// the match into a SwitchStmt with an arm per variant, Opt/Inline.cpp may
// copy that into every caller, SSA/Convert.cpp turns each arm into a block,
// and the vectorizer clones the block into every variant of its function. A
// directive such as `intersect.skip(Shape.Sphere)` (ir::ArmCursors) has to
// find that block wherever it has got to, and a block's name cannot do it:
// `!case_14` says nothing, and every clone renames it. So the arm carries its
// provenance -- what it is, and where it was written -- and whatever copies
// the arm copies that with it.
//
// Interned: one record per distinct provenance, so a Provenance is one
// pointer, an undefined one is a null one, comparing two is comparing
// pointers, and a node with none carries a word. The records live for the
// compile.
class Provenance {
public:
    enum class Kind : uint8_t {
        // An arm of a `match` on a variant type: the arm taking `variant` of
        // the `adt`, in function `func`.
        MatchArm,
        // An arm of an `if` written in function `func`: either arm, and the
        // code after an `if` whose one arm returns, which is its other arm.
        // Named by the function alone, since an `if` has no name of its own:
        // `trace.skip(intersectsp_ray_tri)` wagers on every arm written in
        // that helper, wherever inlining carried it (see ir::ArmCursors).
        IfArm,
    };

    Provenance() = default;

    static Provenance match_arm(const std::string &func, const std::string &adt,
                                const std::string &variant);
    static Provenance if_arm(const std::string &func);

    bool defined() const { return data != nullptr; }
    Kind kind() const;
    // The function the construct was written in, before inlining and
    // specialization moved and copied it.
    const std::string &func() const;
    const std::string &adt() const;
    const std::string &variant() const;

    // Does a schedule's dotted name pick this out? `Shape` names every arm
    // of every match on a Shape; `Shape.Disc` names the arms taking that
    // variant. An `if`'s arm has no such name and matches none; an
    // undefined provenance matches nothing.
    bool matches(const std::vector<std::string> &names) const;

    // `Shape.Disc`: the way a schedule spells it, and `if` for an `if`'s arm.
    // Printing adds the function: `Shape.Disc of area`.
    std::string str() const;

    friend bool operator==(const Provenance &a, const Provenance &b) {
        return a.data == b.data;
    }
    friend bool operator!=(const Provenance &a, const Provenance &b) {
        return a.data != b.data;
    }
    friend bool operator<(const Provenance &a, const Provenance &b) {
        return a.data < b.data;
    }
    friend std::ostream &operator<<(std::ostream &os, const Provenance &p);

private:
    struct Data {
        Kind kind;
        std::string func;
        std::string adt;
        std::string variant;
    };
    const Data *data = nullptr;
};

} // namespace ir
} // namespace bonsai
