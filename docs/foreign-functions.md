# Foreign types and functions

A bonsai program sometimes has to use something it cannot describe: a data
structure in a format and a library the program does not own. NanoVDB is the
case that brought this up. pbrt's `NanoVDBMedium` reads a sparse voxel grid
through NanoVDB's own accessor, never looks at the grid's bytes, uploads the
buffer verbatim to the GPU and calls the same accessor there. In bonsai's
terms that grid is not a set with a layout -- the layout would be a
re-implementation of a library we did not write -- and it is not a program
value either. It is an opaque object with an interface. This note is how such
an object is declared, what the program may do with it, how a call to its
interface is compiled on each target, and what the mechanism gives up.

## The declarations

```
extern element NanoVDBGrid;
extern func nanovdb_grid(bytes : array[u8], at : u32) -> NanoVDBGrid;
extern func nanovdb_value(g : NanoVDBGrid, i : i32, j : i32, k : i32) -> f32;
```

`extern element Name;` declares a **foreign type** (`ir::Foreign_t`). It has
no fields and no definition on the bonsai side; it is one pointer-sized word
whose meaning is the implementation's -- an address, a handle, whatever the
library keeps there. The program never reads the word.

`extern func name(params) -> R;` declares a **foreign function**
(`ir::Function::Attribute::foreign`): a signature with no body. It is called
with the C calling convention under its own name, and defined by something
outside the program. The parser keeps these in `Program::foreign_funcs`,
apart from `Program::funcs`, so that no pass over the program's functions
meets a function without a body; a call to one is an ordinary `Call` whose
callee the passes do not find and leave alone.

What a signature may carry is what has one C meaning on every target:

- a scalar (`i32`, `u64`, `f32`, `bool`, ...), as itself;
- a foreign type, as a pointer;
- as a parameter, `array[T]` of scalars, as the address of its elements.

Not a struct (its layout would have to be agreed field by field), not a short
vector (its ABI differs by target), not `mut` (the function would be handed
the program's storage to write, with nothing checking what it does; return
the value instead), no defaults, no interfaces. The return is a scalar, a
foreign type or `void`. A `bool` is declared zero-extended, as clang
declares one.

## What the program may do with a foreign value

Hold it in a local, pass it to a function (bonsai or foreign), return it. That
is all. A foreign value is **not stored**: it cannot be a field of an element,
an element of an array or a set, an extern, or a layout member. An element
is laid out, copied into pools, uploaded to a device; a handle the library
owns, bound to one address space, cannot be. The pattern is to store what
identifies the value -- an offset into an extern byte array -- and make the
value where it is used, with a foreign function that takes the bytes and
the offset. `nanovdb_grid` above is that function. The parser refuses the
other placements with an error that says so (tests/bonsai/error/foreign-*).

Only a foreign function makes a foreign value; there is no literal and no
cast to one.

## Compilation

Every backend declares each foreign function in its module under the
program's own name, with external linkage and the C types of its signature
(`CodeGen_LLVM::declare_foreign_function`; the PTX generator does the same
in `begin`). A call is then a direct call, found by name as any call is.
Nothing in the module defines the function. What does:

**The driver's link (host only).** A host module may leave the function an
external symbol, as a C program leaves a library's. The C++ backend's
generated header declares every foreign type as a typedef of `const void *`
and every foreign function as an `extern "C"` prototype, so a driver that
includes the header and defines the function against the wrong signature
fails to compile rather than running wrong
(tests/bonsai/correctness/cpp/foreign.bonsai, main_foreign.cpp).

**`--link <file>`, repeatable** (`CompilerOptions::link_files`). An
implementation linked into the generated module *before it is optimized*,
so that it is inlined where it is called, the way CUDA's libdevice is linked
into a kernel (`CodeGen_PTX::link_libdevice`):

- LLVM bitcode or textual IR (`.bc`, `.ll`): `llvm::Linker` with
  `LinkOnlyNeeded`, so only what the program calls comes in, and everything
  that came in is made internal, so it is folded into its callers and
  dropped. A file links into the module whose target matches its own -- a
  host file into the host module, an `nvptx64` file into the device module
  -- and is skipped by the other, so one command line carries both. Written
  from C or C++ with `clang++ -emit-llvm -c`; the tests carry a hand-written
  `.ll` (tests/bonsai/backends/llvm/foreign-impl.ll). With it,
  foreign-linked.bonsai's `sample` is two loads and a multiply and no call
  remains (compare foreign.bonsai, which keeps the `declare`s).
- PTX (`.ptx`), device module only: spliced into the module's own PTX text
  right after its header, ahead of every kernel (ptxas reads once through
  and a call wants its callee above it), its `.version`, `.target` and
  `.address_size` lines dropped, and the module's `.extern .func`
  declarations of the functions it defines dropped too, since PTX refuses a
  name both declared `.extern` and defined
  (`CodeGen_PTX::append_linked_ptx`). What `nvcc -ptx -rdc=true` makes of a
  `__device__` shim. ptxas then folds the function into the kernel as it
  folds every device function. This is the route when clang cannot compile
  the library's header for the device, which is the case for NanoVDB's under
  this machine's CUDA (tests/bonsai/backends/ptx/foreign.bonsai).

A device module that calls a foreign function and has nothing defining it
after the bitcode link is an error, with the function's name and the two
ways to supply it; if PTX was given, ptxas is the one to say.

## What a foreign call is to the compiler

A black box with the shape of its signature. The scheduler cannot defer it,
stage it, or see through it, and the vectorizer cannot widen it: a call in a
vectorized region with a lane's arguments is refused
(tests/bonsai/error/foreign-vectorized.bonsai), as ispc refuses a call to an
external function with varying arguments. A call whose arguments are uniform
across the gang is made once, as any uniform call is. The compiler treats a
foreign call as it treats a call to any function it cannot inline; it does
not assume it is pure, and it does not assume it writes anything the program
can see, since it can be handed nothing the program writes.

For the GPU in particular, a foreign call inside a kernel is a real call
until ptxas inlines it, and the device-side implementation is the shim's
own code: the program's `--fast-math` and register cap apply to the module,
not to PTX appended to it.

## The instance: NanoVDB in apps/pbrt

The medium declares the two functions above and an extern byte array holding
the grid buffers the converter copied out of the `.nvdb` files, each medium
arm carrying its grids' offsets. A density lookup makes the grid handle from
the bytes and the offset, converts the point into the grid's index space with
the map the converter read from the grid's header, and makes eight calls to
`nanovdb_value` for the trilinear filter, in NanoVDB's own convention. The
implementation is apps/pbrt/nanovdb_shim.cpp: two `extern "C"` functions over
NanoVDB's header, compiled into the driver and to host bitcode for the CPU
(`--link nanovdb_shim.bc`), and by nvcc to PTX for the GPU
(`--link nanovdb_shim.ptx`). The majorant grid pbrt builds over the voxels at
load is built by the converter, which links pbrt and its NanoVDB, exactly as
pbrt builds it; the renderer marches it with the same DDA the other grid
media use. See apps/pbrt/PLAN.md, "NanoVDB".

## Tests

- tests/bonsai/parsing/foreign.bonsai: the two declarations and a call,
  printed back.
- tests/bonsai/error/foreign-field, foreign-array, foreign-param,
  foreign-vectorized: the placements and signatures that are refused, and
  the vectorized call.
- tests/bonsai/backends/llvm/foreign.bonsai and foreign-linked.bonsai: the
  declared call, and the same program with foreign-impl.ll linked and
  inlined.
- tests/bonsai/backends/ptx/foreign.bonsai: a kernel's call with
  foreign-impl.ptx appended.
- tests/bonsai/correctness/llvm/foreign.bonsai: run under the JIT with the
  `.ll` linked; tests/bonsai/correctness/cpp/foreign.bonsai: run against a
  driver that defines the functions in C++.
