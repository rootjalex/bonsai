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
        os << "func " << name << "(";
        bool first = true;
        for (const auto &arg : func->args) {
            if (!first) {
                os << ",";
            }
            first = false;
            os << arg.name;
            if (arg.type.defined()) {
                os << " : " << arg.type;
            }
            if (arg.default_value.defined()) {
                os << " = " << arg.default_value;
            }
        }
        os << ") -> " << func->ret_type << " {\n" << func->body << "}\n\n";
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
