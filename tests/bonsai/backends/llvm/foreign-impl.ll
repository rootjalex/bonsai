; The implementation of foreign.bonsai's two foreign functions, as LLVM IR
; for the host: what `--link` takes (CompilerOptions::link_files). An
; implementation is ordinarily compiled from C or C++ -- apps/pbrt's
; nanovdb_shim.cpp, with `clang++ -emit-llvm -c` -- and lands here as the
; same thing; this one is written out so that the test owns it.
;
; `grid_of` is the address of the `at`th float; `grid_value` reads the
; `i`th float from there and doubles it. No target triple: the file is for
; whichever host module links it, which sets the layout and the triple to
; its own.

define ptr @grid_of(ptr %bytes, i32 %at) {
  %p = getelementptr float, ptr %bytes, i32 %at
  ret ptr %p
}

define float @grid_value(ptr %g, i32 %i) {
  %p = getelementptr float, ptr %g, i32 %i
  %v = load float, ptr %p
  %r = fmul float %v, 2.0
  ret float %r
}
