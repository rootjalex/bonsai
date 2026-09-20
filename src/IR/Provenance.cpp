#include "IR/Provenance.h"

#include "Error.h"

#include <map>
#include <memory>
#include <mutex>
#include <ostream>
#include <tuple>

namespace bonsai {
namespace ir {

Provenance Provenance::match_arm(const std::string &func,
                                 const std::string &adt,
                                 const std::string &variant) {
    internal_assert(!adt.empty() && !variant.empty())
        << "A match arm's provenance needs the type and the variant";
    // The one record for each provenance made so far. Records are never
    // freed: a Provenance is a pointer into this table, and there are as many
    // distinct ones as the program has match arms.
    static std::mutex lock;
    static std::map<std::tuple<std::string, std::string, std::string>,
                    std::unique_ptr<Data>>
        records;
    const std::lock_guard<std::mutex> guard(lock);
    auto &record = records[std::make_tuple(func, adt, variant)];
    if (record == nullptr) {
        record = std::make_unique<Data>(
            Data{Kind::MatchArm, func, adt, variant});
    }
    Provenance p;
    p.data = record.get();
    return p;
}

Provenance::Kind Provenance::kind() const {
    internal_assert(defined()) << "kind() of an undefined provenance";
    return data->kind;
}

const std::string &Provenance::func() const {
    internal_assert(defined()) << "func() of an undefined provenance";
    return data->func;
}

const std::string &Provenance::adt() const {
    internal_assert(defined()) << "adt() of an undefined provenance";
    return data->adt;
}

const std::string &Provenance::variant() const {
    internal_assert(defined()) << "variant() of an undefined provenance";
    return data->variant;
}

bool Provenance::matches(const std::vector<std::string> &names) const {
    if (!defined() || names.empty() || names.size() > 2) {
        return false;
    }
    return names[0] == data->adt &&
           (names.size() == 1 || names[1] == data->variant);
}

std::string Provenance::str() const {
    if (!defined()) {
        return "";
    }
    return data->adt + "." + data->variant;
}

// `Shape.Ring of area`: the arm and the function it was written in, since a
// dump reads it where inlining and specialization have put it.
std::ostream &operator<<(std::ostream &os, const Provenance &p) {
    if (!p.defined()) {
        return os;
    }
    return os << p.str() << " of " << p.func();
}

} // namespace ir
} // namespace bonsai
