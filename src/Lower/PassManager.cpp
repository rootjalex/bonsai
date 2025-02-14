#include "Lower/PassManager.h"

#include "IR/Mutator.h"

#include "CompilerOptions.h"
#include "Error.h"
#include "Utils.h"

#include <memory>
#include <span>
#include <string>

namespace bonsai {
namespace lower {

void PassManager::register_alias(std::string_view alias,
                                 std::span<std::unique_ptr<Pass>> passes) {
    auto [it, succeeded] = registered_aliases.try_emplace(
        std::string(alias), std::vector<std::string>{});
    internal_assert(succeeded) << "alias already inserted: " << alias;
    auto &ps = it->second;
    for (auto &p : passes) {
        const std::string &name = p->name();
        ps.push_back(name);
        // When registering an alias we may see the same pass twice; that's ok.
        registered_passes.try_emplace(name, std::move(p));
    }
}

// Returns the pass with this name.
Pass *PassManager::get_pass(std::string_view name) const {
    auto it = registered_passes.find(std::string(name));
    if (it == registered_passes.end()) {
        internal_error << "pass not found: " << name;
    }
    return it->second.get();
}

// Returns a list of passes for this alias.
std::vector<Pass *>
PassManager::get_alias_passes(std::string_view alias) const {
    auto it = registered_aliases.find(std::string(alias));
    if (it == registered_aliases.end()) {
        internal_error << "alias not found: " << alias;
    }
    std::vector<Pass *> passes;
    for (const std::string &name : it->second) {
        auto it = registered_passes.find(name);
        if (it == registered_passes.end()) {
            internal_error << "pass not found: " << name;
        }
        passes.push_back(it->second.get());
    }
    return passes;
}

bool PassManager::has_alias(std::string_view alias) const {
    return registered_aliases.contains(std::string(alias));
}

} // namespace lower
} // namespace bonsai
