#include "Error.h"

namespace bonsai {

std::ostream &operator<<(std::ostream &os, const Error &error) {
    os << error.message();
    return os;
}

} // namespace bonsai
