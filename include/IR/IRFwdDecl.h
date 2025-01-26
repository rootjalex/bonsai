#pragma once

namespace bonsai {
namespace ir {

// Types
struct Type;
struct Int_t;
struct UInt_t;
struct Float_t;
struct Bool_t;
struct Ptr_t;
struct Vector_t;
struct Struct_t;
struct Tuple_t;
struct Option_t;
struct Set_t;
struct Function_t;

// Exprs
struct Expr;
struct IntImm;
struct UIntImm;
struct FloatImm;
struct BoolImm;
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
struct Intrinsic;
struct Lambda;
struct GeomOp;
struct SetOp;
struct Call;

// Stmts
struct Stmt;
struct Return;
struct Store;
struct LetStmt;
struct IfElse;
struct Sequence;
struct Assign;
struct Accumulate;

} // namespace ir
} // namespace bonsai
