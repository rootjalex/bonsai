#include "IR/Program.h"
#include "IR/Printer.h"

#include <stdexcept>

namespace bonsai {
namespace ir {

void Program::dump(std::ostream &os) const {
    Printer printer(os, /*verbose=*/true);
    for (const auto &[name, type] : types) {
        os << "type " << name << " = ";
        printer.print(type);
        os << "\n";
    }
    if (!types.empty()) {
        os << std::endl;
    }
    for (const auto &[name, type] : externs) {
        os << "extern " << name << " : ";
        printer.print(type);
        os << "\n";
    }
    if (!externs.empty()) {
        os << std::endl;
    }
    for (const auto &[name, func] : funcs) {
        os << *func << "\n\n";
    }
    if (!funcs.empty()) {
        os << std::endl;
    }
    for (const auto &[target, schedule] : schedules) {
        os << "schedule " << target << "{\n";
        os << schedule << "}";
    }
    if (!schedules.empty()) {
        os << std::endl;
    }
}

} // namespace ir
} // namespace bonsai
