#pragma once

namespace bonsai {
namespace ir {

// Types
struct Type;
struct Void_t;
struct Int_t;
struct UInt_t;
struct Index_t;
struct Float_t;
struct Bool_t;
struct Ptr_t;
struct Ref_t;
struct Vector_t;
struct Struct_t;
struct Tuple_t;
struct Array_t;
struct Option_t;
struct Set_t;
struct Function_t;
struct Generic_t;
struct BVH_t;

// Interfaces
struct Interface;
struct IEmpty;
struct IFloat;
struct IVector;

// Exprs
struct Expr;
struct IntImm;
struct UIntImm;
struct FloatImm;
struct BoolImm;
struct VecImm;
struct Infinity;
struct Var;
struct BinOp;
struct UnOp;
struct Select;
struct Cast;
struct Broadcast;
struct VectorReduce;
struct VectorShuffle;
struct Ramp;
struct Extract;
struct Build;
struct Access;
struct Unwrap;
struct Intrinsic;
struct Lambda;
struct GeomOp;
struct SetOp;
struct Call;
struct Instantiate;

// Stmts
struct Stmt;
struct CallStmt;
struct Print;
struct Return;
struct Store;
struct LetStmt;
struct IfElse;
struct DoWhile;
struct Sequence;
struct Assign;
struct Accumulate;
struct Allocate;
struct Label;
struct RecLoop;
struct Match;
struct Yield;
struct Scan;
struct YieldFrom;
struct ForAll;
struct ForEach;
struct Continue;

// Layouts
struct Name;
struct Pad;
struct Split;
struct Chain;
struct Group;
struct Materialize;

struct WriteLoc;

} // namespace ir
} // namespace bonsai
