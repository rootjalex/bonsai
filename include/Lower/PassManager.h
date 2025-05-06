#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

#include <memory>
#include <span>
#include <string>
#include <unordered_map>

namespace bonsai {
namespace lower {

// Pass manager for Bonsai lowering.
class PassManager {
  public:
    template <typename T>
    void register_pass() {
        std::unique_ptr<Pass> pass = std::make_unique<T>();
        const std::string &name = pass->name();
        auto [_, succeeded] =
            registered_passes.try_emplace(name, std::move(pass));
        internal_assert(succeeded) << "pass already inserted: " << name;
    }

    void register_alias(std::string_view alias,
                        std::span<std::unique_ptr<Pass>> passes);

    // Returns the pass with this name.
    Pass *get_pass(std::string_view name) const;

    // Returns a list of passes for this alias.
    std::vector<Pass *> get_alias_passes(std::string_view alias) const;

    // Returns whether the pass manager has register the following alias.
    bool has_alias(std::string_view alias) const;

  private:
    // A mapping from pass name to its respective pass.
    std::unordered_map<std::string, std::unique_ptr<Pass>> registered_passes;
    // A mapping from an alias to a list of pass names.
    std::unordered_map<std::string, std::vector<std::string>>
        registered_aliases;
};

} // namespace lower
} // namespace bonsai
