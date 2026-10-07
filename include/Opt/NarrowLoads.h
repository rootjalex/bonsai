#pragma once

#include "CompilerOptions.h"
#include "IR/Program.h"
#include "Lower/Pass.h"

namespace bonsai {
namespace opt {

// Dead-field elimination on aggregate loads.
//
// A `let x = *p` of a struct whose every later use is a field read
// `x.f` becomes one let per field actually read -- `x$f = *&(*p).f`,
// at the same statement -- and the fields nothing reads are never
// loaded at all. The loads stay exactly where the aggregate load was,
// so nothing moves in time and no invariance argument is needed: the
// same memory is read at the same point, just only the bytes the
// program goes on to use. The rule reapplies to a narrowed field that
// is itself a struct, one level a round.
//
// What it is for: a query's hit program binds its element by loading
// the whole stored struct, and the filter body -- after inlining --
// reads two fields of eight. LLVM keeps the aggregate load whole;
// the ruling is that this compiler eliminates the dead fields itself.
// Running just before Opt/Sink composes the two: the narrowed field
// loads then sink individually into the branch arms that alone read
// them.
//
// The same problem crosses a merge: the inliner gives a value-returning
// match a result temp, every arm stores the WHOLE struct into it, and
// the reads after the merge want two fields -- the dead fields stay
// alive through every arm because the store is whole. The second
// pattern is scalar replacement of aggregates (Muchnick 12.2) on that
// temp: one temp per field read, each arm's store split field-wise so
// the unread fields' producing chains die arm by arm under DCE, each
// read renamed. Same statement, same values; a store's value must be
// pure (ValueClass) since it is read once per split field.
class NarrowLoads : public lower::Pass {
  public:
    const std::string name() const override { return "narrow-loads"; }

    ir::FuncMap run(ir::FuncMap funcs,
                    const CompilerOptions &options) const override;
};

} // namespace opt
} // namespace bonsai
