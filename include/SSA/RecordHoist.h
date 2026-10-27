#ifndef BONSAI_SSA_RECORD_HOIST_H
#define BONSAI_SSA_RECORD_HOIST_H

// What a hit program can know from its hit-group record alone is computed
// once per record, by the driver, and read off the record -- pbrt's
// TriangleMeshRecord holds the mesh's own pointers and attributes, and its
// closest-hit program goes from the record to the mesh's index array in one
// load; ours held the mesh's *index* and loaded the mesh's row first.
//
// The record (`_RTRecord`, Lower/Trees.cpp's record_type_of) carries the
// holder element's fields the hit programs read. A value a program computes
// from those fields and from storage the program never writes -- a row of an
// extern array at an index that is a record field, `meshes[rec._1_mesh]`,
// its `first_index` -- depends on nothing of the hit: it is the same for
// every hit on that build input. Partial evaluation with the record as the
// static input (Jones, Gomard & Sestoft, "Partial Evaluation and Automatic
// Program Generation", 1993, ch. 4-5: a binding-time analysis, then a
// residual program and a generating extension): the binding times here are
// three -- a value of the hit (the primitive index, the ray, what depends on
// them), a value of the record, a value of the launch (a parameter, a
// constant) -- and the residual program is the hit program with every
// record-time *load* replaced by a read of a new record field, while the
// generating extension is the record's filler (`bonsai_rt_record_fill`,
// Lower/Trees.cpp's add_record_fill), extended to compute the field in the
// arm whose fields the load's address reads.
//
// Only loads are hoisted, never arithmetic: a load is the latency the hit
// waits for, and a sum of two record fields costs less in the program than a
// third field would in every record. A load's address at record time means
// the array is read-only storage (an extern, `const T[]`) and the index a
// record-time value; a value of the launch alone (a parameter's field, a
// constant) is not hoisted either, since nothing of the record varies it.
//
// The analysis sees through the calls a program makes: a call whose
// argument carries a record-time part -- `triangle_of(Triangle{rec.mesh,
// idx})` -- is inlined first (inline_record_callees, SSA/Rewrite.h's
// inline_call), so that the callee's loads at the record's index are the
// program's own.
//
// The record's type grows by the hoisted fields, named `_<variant>_h<n>`
// after the arm whose fields the address reads; every read of the record
// (`rt_sbt_record()`) is retyped, the filler's record built per arm gets the
// field's value in that arm and a zero in the others, and the filler takes
// the arrays the loads read as further parameters, by the names the program
// knows them by. runtime/bonsai_optix.h and the drivers see only a longer
// record (BONSAI_OPTIX_RECORD_BYTES).

#include "IR/Program.h"
#include "SSA/SSA.h"

#include <map>
#include <memory>
#include <string>

namespace bonsai {
namespace ir {
namespace ssa {

using RecordFuncMap = std::map<std::string, std::shared_ptr<Function>>;

// In every function that reads the hit-group record, inlines each call whose
// arguments carry a record-time part, until none is left. Before the
// promotion of `mut` locals (the copies bring their own allocas, promoted
// with the rest).
void inline_record_callees(RecordFuncMap &funcs);

// Hoists the record-time loads of every function that reads the record into
// the record, extending the type, the reads and the filler (above). After the
// simplification that follows the promotion of allocas, so that a struct a
// callee built and the program took apart is its fields again.
// `program` is where the record's type lives (ir::Program::types).
void hoist_record_loads(RecordFuncMap &funcs, ir::Program &program);

} // namespace ssa
} // namespace ir
} // namespace bonsai

#endif
