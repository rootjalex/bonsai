#include "Lower/Layouts.h"

#include "IR/Mutator.h"
#include "IR/Printer.h"
#include "IR/Visitor.h"

#include "Error.h"

namespace bonsai {
namespace lower {

namespace {

// ensures unique names in lowering.
static size_t name_counter = 0;
static size_t pad_counter = 0;
static size_t split_counter = 0;
static size_t group_counter = 0;

std::string unique_struct_name(std::string base) {
    return "?" + base + "_layout" + std::to_string(name_counter++);
}

std::string unique_pad_name() { return "?pad" + std::to_string(pad_counter++); }

std::string unique_split_name(std::string base) {
    return "?split_" + base + std::to_string(split_counter++);
}

std::string unique_group_name() {
    return "?group" + std::to_string(group_counter++);
}

std::vector<ir::Type> layout_to_structs(std::string base,
                                        const ir::Layout &layout) {
    std::vector<ir::Type> rets;
    if (const ir::Chain *chain = layout.as<ir::Chain>()) {
        ir::Struct_t::Map fields;
        for (const auto &l : chain->layouts) {
            switch (l.node_type()) {
            case ir::IRLayoutEnum::Name: {
                const ir::Name *node = l.as<ir::Name>();
                fields.emplace_back(node->name, node->type);
                break;
            }
            case ir::IRLayoutEnum::Pad: {
                const ir::Pad *node = l.as<ir::Pad>();
                ir::Type pad_type = ir::UInt_t::make(node->bits);
                fields.emplace_back(unique_pad_name(), std::move(pad_type));
                break;
            }
            case ir::IRLayoutEnum::Group: {
                const ir::Group *node = l.as<ir::Group>();
                std::string new_base =
                    node->name.empty() ? "" : base + "_" + node->name;
                std::vector<ir::Type> rec =
                    layout_to_structs(new_base, node->inner);
                internal_assert(!rec.empty());
                ir::Type base_t = rec.back();
                ir::Type group_t =
                    ir::Array_t::make(std::move(base_t), node->size);
                // Add to rets.
                rets.insert(rets.end(), std::make_move_iterator(rec.begin()),
                            std::make_move_iterator(rec.end()));
                std::string field_name =
                    new_base.empty() ? unique_group_name() : new_base;
                // push back new field type.
                fields.emplace_back(std::move(field_name), std::move(group_t));
                break;
            }
            case ir::IRLayoutEnum::Split: {
                const ir::Split *node = l.as<ir::Split>();
                // Store as vector of bytes, load and reinterpret to proper
                // type.
                const uint64_t bits = l.bits();
                internal_assert(bits % 8 == 0)
                    << "Split is not byte-aligned: " << l;
                static const ir::Type u8 = ir::UInt_t::make(8);
                ir::Type byte_vec = ir::Vector_t::make(u8, bits / 8);
                fields.emplace_back(unique_split_name(node->field),
                                    std::move(byte_vec));
                break;
            }
            default: {
                internal_error << "Handle layout in Chain lowering: " << l;
            }
            }
        }
        rets.push_back(ir::Struct_t::make(unique_struct_name(std::move(base)),
                                          std::move(fields)));
        return rets;
    }
    internal_error << "Handle layout conversion for: " << layout;
}

} // namespace

ir::Program LowerLayouts::run(ir::Program program) const {
    if (program.schedules.empty()) {
        return program;
    }
    internal_assert(program.schedules.size() == 1)
        << "TODO: support selecting a schedule target!\n";

    ir::LayoutMap tree_layouts =
        std::move(program.schedules[ir::Target::Host].tree_layouts);

    if (tree_layouts.empty()) {
        return program;
    }

    for (const auto &[name, layout] : tree_layouts) {
        auto struct_ts = layout_to_structs(name, layout);
        std::cout << "Lowered layout: " << layout << " to structs:\n";
        ir::Printer printer(std::cout, /*verbose=*/true);
        for (const auto &type : struct_ts) {
            printer.print(type);
            std::cout << "\n\n";
        }
    }

    internal_error << "TODO: finish implementing layout lowering!";
    return program;
}

} // namespace lower
} // namespace bonsai
