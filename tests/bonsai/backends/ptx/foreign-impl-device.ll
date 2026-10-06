; The implementation of foreign-bitcode.bonsai's `grid_value` as nvptx64
; LLVM IR: the `i`th float at the address `g` holds, doubled -- what clang
; `--target=nvptx64-nvidia-cuda -O2 -emit-llvm` writes for the same C++ shim
; foreign-impl.ptx was made from, shortened by hand (a text .ll rather than
; the .bc clang writes, so the file reads as source; parseIRFile takes
; either). `--link` routes it into the device module by this triple
; (link_foreign_implementations), where the definition is internalized and
; marked always-inline like every other device function.
target datalayout = "e-p6:32:32-i64:64-i128:128-i256:256-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define float @grid_value(ptr %g, i32 %i) {
entry:
  %idx = sext i32 %i to i64
  %p = getelementptr inbounds float, ptr %g, i64 %idx
  %v = load float, ptr %p, align 4
  %d = fadd float %v, %v
  ret float %d
}
