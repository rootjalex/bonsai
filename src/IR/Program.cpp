#include "IR/Program.h"
#include "IR/Printer.h"

#include <stdexcept>

namespace bonsai {
namespace ir {

void Program::dump(std::ostream &os) const {
    for (const auto &[name, type] : types) {
        os << "type " << name << " = " << type << "\n";
    }
    os << std::endl;
    for (const auto &[name, type] : externs) {
        os << "extern " << name << " : " << type << "\n";
    }
    os << std::endl;
    for (const auto &[name, func] : funcs) {
        os << *func << "\n\n";
    }
    os << std::endl;
    if (main_body.defined()) {
        os << "func main() {";
        os << main_body;
        os << "}";
    }
}

} // namespace ir
} // namespace bonsai
