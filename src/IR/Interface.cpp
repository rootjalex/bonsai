#include "IR/Interface.h"

#include "IR/Printer.h"

#include <utility>

namespace bonsai {
namespace ir {

bool Interface::is_numeric() const {
    return is<IFloat>() || (is<IVector>() && as<IVector>()->etype.is_numeric());
}

Interface IEmpty::make() {
    static Interface iempty = new IEmpty;
    return iempty;
}

Interface IFloat::make() {
    static Interface ifloat = new IFloat;
    return ifloat;
}

Interface IVector::make(Interface etype) {
    internal_assert(etype.defined())
        << "Cannot make IVector with undefined etype";
    IVector *node = new IVector;
    node->etype = std::move(etype);
    return node;
}

} // namespace ir
} // namespace bonsai
