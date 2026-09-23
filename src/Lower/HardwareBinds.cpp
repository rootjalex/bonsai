#include "Lower/HardwareBinds.h"

#include "IR/Equality.h"
#include "IR/Expr.h"
#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Schedule.h"
#include "IR/Stmt.h"
#include "Error.h"

#include <map>
#include <string>

namespace bonsai {
namespace lower {

namespace {

// The lambda's parameter names replaced by the function's: `t` for whatever
// the schedule called its first parameter, and so on.
struct RenameVars : public ir::Mutator {
    explicit RenameVars(const std::map<std::string, std::string> &renames)
        : renames(renames) {}

    ir::Expr visit(const ir::Var *node) override {
        const auto it = renames.find(node->name);
        if (it == renames.end()) {
            return node;
        }
        return ir::Var::make(node->type, it->second);
    }

    const std::map<std::string, std::string> &renames;
};

} // namespace

ir::Program LowerHardwareBinds::run(ir::Program program,
                                    const CompilerOptions &) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";
    const ir::TransformMap &transforms =
        program.schedules[ir::Target::Host].func_transforms;

    for (const auto &[name, ts] : transforms) {
        for (const ir::Transform &t : ts) {
            const auto *bind = std::get_if<ir::Bind>(&t);
            if (bind == nullptr || !bind->lambda.defined()) {
                continue;
            }
            const auto fiter = program.funcs.find(name);
            internal_assert(fiter != program.funcs.end())
                << name << ".bind(" << to_string(bind->resource)
                << ", ...): no function " << name;
            ir::Function &func = *fiter->second;
            const ir::Lambda *lambda = bind->lambda.as<ir::Lambda>();
            internal_assert(lambda != nullptr)
                << name << ".bind(" << to_string(bind->resource)
                << ", ...): the second argument is not a lambda: "
                << bind->lambda;

            // The lambda is the function, so it takes what the function
            // takes and gives what the function gives.
            internal_assert(lambda->args.size() == func.args.size())
                << name << ".bind(" << to_string(bind->resource)
                << ", ...): the lambda takes " << lambda->args.size()
                << " parameters and " << name << " takes "
                << func.args.size();
            std::map<std::string, std::string> renames;
            for (size_t k = 0; k < func.args.size(); k++) {
                const ir::TypedVar &given = lambda->args[k];
                const ir::Function::Argument &wanted = func.args[k];
                internal_assert(ir::equals(given.type, wanted.type))
                    << name << ".bind(" << to_string(bind->resource)
                    << ", ...): the lambda's parameter " << k << " (`"
                    << given.name << "`) is a " << given.type << " where "
                    << name << "'s is a " << wanted.type;
                if (given.name != wanted.name) {
                    renames[given.name] = wanted.name;
                }
            }
            internal_assert(ir::equals(lambda->value.type(), func.ret_type))
                << name << ".bind(" << to_string(bind->resource)
                << ", ...): the lambda gives a " << lambda->value.type()
                << " where " << name << " returns a " << func.ret_type;

            ir::Expr value = RenameVars(renames).mutate(lambda->value);
            func.body = ir::Return::make(std::move(value));
        }
    }
    return program;
}

} // namespace lower
} // namespace bonsai
