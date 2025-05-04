; ModuleID = 'bonsai_module'
source_filename = "bonsai_module"

%struct.Ray = type { <3 x float>, <3 x float> }

; Function Attrs: nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite)
define noalias noundef ptr @_traverse_array0(i32 noundef %width, i32 noundef %height) local_unnamed_addr #0 {
_traverse_array0_entry:
  %0 = mul i32 %width, 3
  %1 = mul i32 %0, %height
  %size64 = zext i32 %1 to i64
  %mallocsize = shl nuw nsw i64 %size64, 2
  %_untyped = tail call ptr @malloc(i64 %mallocsize)
  %2 = icmp sgt i32 %height, 0
  br i1 %2, label %_i00_for.preheader, label %_i00_for_end, !prof !0

_i00_for.preheader:                               ; preds = %_traverse_array0_entry
  %3 = icmp sgt i32 %width, 0
  %4 = sitofp i32 %width to float
  %5 = fdiv float %4, 0x3FFC71C720000000
  %6 = fptosi float %5 to i32
  %7 = tail call i32 @llvm.smax.i32(i32 %6, i32 1)
  %8 = uitofp nneg i32 %7 to float
  %9 = fdiv float %4, %8
  %10 = fmul float %9, 2.000000e+00
  %11 = insertelement <3 x float> <float poison, float 0.000000e+00, float 0.000000e+00>, float %10, i64 0
  %.splatinsert.i = insertelement <3 x float> poison, float %4, i64 0
  %.splat.i = shufflevector <3 x float> %.splatinsert.i, <3 x float> poison, <3 x i32> zeroinitializer
  %12 = fdiv <3 x float> %11, %.splat.i
  %.splatinsert4.i = insertelement <3 x float> poison, float %8, i64 0
  %.splat5.i = shufflevector <3 x float> %.splatinsert4.i, <3 x float> poison, <3 x i32> zeroinitializer
  %13 = fdiv <3 x float> <float 0.000000e+00, float -2.000000e+00, float 0.000000e+00>, %.splat5.i
  %.scalar.i = fmul float %10, 5.000000e-01
  %.scalar13.i = fsub float 0.000000e+00, %.scalar.i
  %14 = insertelement <3 x float> <float poison, float 1.000000e+00, float -1.000000e+00>, float %.scalar13.i, i64 0
  %15 = fadd <3 x float> %13, %12
  %16 = fmul <3 x float> %15, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %17 = fadd <3 x float> %14, %16
  %18 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>)
  %19 = fadd float %18, -2.500000e-01
  br i1 %3, label %_i00_for.us, label %_i00_for_end, !prof !0

_i00_for.us:                                      ; preds = %_i00_for.preheader, %_i11_for_end.loopexit.us
  %20 = phi i32 [ %75, %_i11_for_end.loopexit.us ], [ 0, %_i00_for.preheader ]
  %21 = uitofp nneg i32 %20 to float
  %.splatinsert8.i.us = insertelement <3 x float> poison, float %21, i64 0
  %.splat9.i.us = shufflevector <3 x float> %.splatinsert8.i.us, <3 x float> poison, <3 x i32> zeroinitializer
  %22 = fmul <3 x float> %13, %.splat9.i.us
  %23 = mul i32 %20, %width
  br label %_i11_for.us

_i11_for.us:                                      ; preds = %_i00_for.us, %pixel.exit.us
  %24 = phi i32 [ %74, %pixel.exit.us ], [ 0, %_i00_for.us ]
  %25 = uitofp nneg i32 %24 to float
  %.splatinsert6.i.us = insertelement <3 x float> poison, float %25, i64 0
  %.splat7.i.us = shufflevector <3 x float> %.splatinsert6.i.us, <3 x float> poison, <3 x i32> zeroinitializer
  %26 = fmul <3 x float> %12, %.splat7.i.us
  %27 = fadd <3 x float> %17, %26
  %28 = fadd <3 x float> %22, %27
  %29 = fmul <3 x float> %28, %28
  %30 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %29)
  %31 = fmul <3 x float> %28, <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>
  %32 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %31)
  %33 = fmul float %32, -2.000000e+00
  %34 = fmul float %33, %33
  %35 = fmul float %30, 4.000000e+00
  %36 = fmul float %19, %35
  %37 = fsub float %34, %36
  %38 = fcmp olt float %37, 0.000000e+00
  br i1 %38, label %next_bb.i.us, label %hit_sphere.exit.i.us

hit_sphere.exit.i.us:                             ; preds = %_i11_for.us
  %39 = fmul float %30, 2.000000e+00
  %40 = fneg float %33
  %41 = tail call float @llvm.sqrt.f32(float %37)
  %42 = fsub float %40, %41
  %43 = fdiv float %42, %39
  %44 = fcmp ogt float %43, 0.000000e+00
  br i1 %44, label %then_bb.i.us, label %next_bb.i.us

then_bb.i.us:                                     ; preds = %hit_sphere.exit.i.us
  %.splatinsert.i.i.us = insertelement <3 x float> poison, float %43, i64 0
  %.splat.i.i.us = shufflevector <3 x float> %.splatinsert.i.i.us, <3 x float> poison, <3 x i32> zeroinitializer
  %45 = fmul <3 x float> %28, %.splat.i.i.us
  %46 = fadd <3 x float> %45, zeroinitializer
  %47 = fadd <3 x float> %46, <float -0.000000e+00, float -0.000000e+00, float 1.000000e+00>
  %48 = fmul <3 x float> %47, %47
  %49 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %48)
  %50 = tail call float @llvm.sqrt.f32(float %49)
  %.splatinsert.i9.us = insertelement <3 x float> poison, float %50, i64 0
  %.splat.i10.us = shufflevector <3 x float> %.splatinsert.i9.us, <3 x float> poison, <3 x i32> zeroinitializer
  %51 = fdiv <3 x float> %47, %.splat.i10.us
  %52 = fadd <3 x float> %51, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %53 = fmul <3 x float> %52, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  br label %pixel.exit.us

next_bb.i.us:                                     ; preds = %hit_sphere.exit.i.us, %_i11_for.us
  %54 = tail call float @llvm.sqrt.f32(float %30)
  %55 = extractelement <3 x float> %28, i64 1
  %56 = fdiv float %55, %54
  %57 = fadd float %56, 1.000000e+00
  %58 = fmul float %57, 5.000000e-01
  %59 = fsub float 1.000000e+00, %58
  %.splatinsert5.i.us = insertelement <3 x float> poison, float %59, i64 0
  %.splat6.i.us = shufflevector <3 x float> %.splatinsert5.i.us, <3 x float> poison, <3 x i32> zeroinitializer
  %.splatinsert7.i.us = insertelement <3 x float> poison, float %58, i64 0
  %.splat8.i.us = shufflevector <3 x float> %.splatinsert7.i.us, <3 x float> poison, <3 x i32> zeroinitializer
  %60 = fmul <3 x float> %.splat8.i.us, <float 5.000000e-01, float 0x3FE6666660000000, float 1.000000e+00>
  %61 = fadd <3 x float> %.splat6.i.us, %60
  br label %pixel.exit.us

pixel.exit.us:                                    ; preds = %next_bb.i.us, %then_bb.i.us
  %common.ret.op.i.us = phi <3 x float> [ %53, %then_bb.i.us ], [ %61, %next_bb.i.us ]
  %62 = fmul <3 x float> %common.ret.op.i.us, <float 0x406FFFF7C0000000, float 0x406FFFF7C0000000, float 0x406FFFF7C0000000>
  %63 = fptosi <3 x float> %62 to <3 x i32>
  %64 = extractelement <3 x i32> %63, i64 0
  %65 = add i32 %24, %23
  %66 = mul i32 %65, 3
  %67 = sext i32 %66 to i64
  %_alloc0_ld1.us = getelementptr inbounds i32, ptr %_untyped, i64 %67
  store i32 %64, ptr %_alloc0_ld1.us, align 4
  %68 = extractelement <3 x i32> %63, i64 1
  %69 = add i32 %66, 1
  %70 = sext i32 %69 to i64
  %_alloc0_ld3.us = getelementptr inbounds i32, ptr %_untyped, i64 %70
  store i32 %68, ptr %_alloc0_ld3.us, align 4
  %71 = extractelement <3 x i32> %63, i64 2
  %72 = add i32 %66, 2
  %73 = sext i32 %72 to i64
  %_alloc0_ld5.us = getelementptr inbounds i32, ptr %_untyped, i64 %73
  store i32 %71, ptr %_alloc0_ld5.us, align 4
  %74 = add nuw nsw i32 %24, 1
  %.not.us = icmp slt i32 %74, %width
  br i1 %.not.us, label %_i11_for.us, label %_i11_for_end.loopexit.us

_i11_for_end.loopexit.us:                         ; preds = %pixel.exit.us
  %75 = add nuw nsw i32 %20, 1
  %.not8.us = icmp slt i32 %75, %height
  br i1 %.not8.us, label %_i00_for.us, label %_i00_for_end

_i00_for_end:                                     ; preds = %_i11_for_end.loopexit.us, %_i00_for.preheader, %_traverse_array0_entry
  ret ptr %_untyped
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define <3 x float> @at(ptr nocapture noundef nonnull readonly %r, float noundef %t) local_unnamed_addr #1 {
at_entry:
  %deref_temp.unpack = load <3 x float>, ptr %r, align 16
  %deref_temp.elt2 = getelementptr inbounds i8, ptr %r, i64 16
  %deref_temp.unpack3 = load <3 x float>, ptr %deref_temp.elt2, align 16
  %.splatinsert = insertelement <3 x float> poison, float %t, i64 0
  %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
  %0 = fmul <3 x float> %.splat, %deref_temp.unpack3
  %1 = fadd <3 x float> %deref_temp.unpack, %0
  ret <3 x float> %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define %struct.Ray @build_ray(i32 noundef %i, i32 noundef %j, i32 noundef %width) local_unnamed_addr #2 {
build_ray_entry:
  %0 = sitofp i32 %width to float
  %1 = fdiv float %0, 0x3FFC71C720000000
  %2 = fptosi float %1 to i32
  %3 = tail call i32 @llvm.smax.i32(i32 %2, i32 1)
  %4 = uitofp nneg i32 %3 to float
  %5 = fdiv float %0, %4
  %6 = fmul float %5, 2.000000e+00
  %7 = insertelement <3 x float> <float poison, float 0.000000e+00, float 0.000000e+00>, float %6, i64 0
  %.splatinsert = insertelement <3 x float> poison, float %0, i64 0
  %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
  %8 = fdiv <3 x float> %7, %.splat
  %.splatinsert4 = insertelement <3 x float> poison, float %4, i64 0
  %.splat5 = shufflevector <3 x float> %.splatinsert4, <3 x float> poison, <3 x i32> zeroinitializer
  %9 = fdiv <3 x float> <float 0.000000e+00, float -2.000000e+00, float 0.000000e+00>, %.splat5
  %.scalar = fmul float %6, 5.000000e-01
  %.scalar13 = fsub float 0.000000e+00, %.scalar
  %10 = insertelement <3 x float> <float poison, float 1.000000e+00, float -1.000000e+00>, float %.scalar13, i64 0
  %11 = fadd <3 x float> %9, %8
  %12 = fmul <3 x float> %11, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  %13 = fadd <3 x float> %10, %12
  %14 = sitofp i32 %i to float
  %.splatinsert6 = insertelement <3 x float> poison, float %14, i64 0
  %.splat7 = shufflevector <3 x float> %.splatinsert6, <3 x float> poison, <3 x i32> zeroinitializer
  %15 = fmul <3 x float> %.splat7, %8
  %16 = fadd <3 x float> %15, %13
  %17 = sitofp i32 %j to float
  %.splatinsert8 = insertelement <3 x float> poison, float %17, i64 0
  %.splat9 = shufflevector <3 x float> %.splatinsert8, <3 x float> poison, <3 x i32> zeroinitializer
  %18 = fmul <3 x float> %.splat9, %9
  %19 = fadd <3 x float> %18, %16
  %20 = insertvalue %struct.Ray { <3 x float> zeroinitializer, <3 x float> undef }, <3 x float> %19, 1
  ret %struct.Ray %20
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define float @hit_sphere(<3 x float> noundef %center, float noundef %radius, ptr nocapture noundef nonnull readonly %r) local_unnamed_addr #1 {
hit_sphere_entry:
  %deref_temp.unpack = load <3 x float>, ptr %r, align 16
  %deref_temp.elt4 = getelementptr inbounds i8, ptr %r, i64 16
  %deref_temp.unpack5 = load <3 x float>, ptr %deref_temp.elt4, align 16
  %0 = fsub <3 x float> %center, %deref_temp.unpack
  %1 = fmul <3 x float> %deref_temp.unpack5, %deref_temp.unpack5
  %2 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %1)
  %3 = fmul <3 x float> %deref_temp.unpack5, %0
  %4 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %3)
  %5 = fmul float %4, -2.000000e+00
  %6 = fmul <3 x float> %0, %0
  %7 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %6)
  %8 = fmul float %radius, %radius
  %9 = fsub float %7, %8
  %10 = fmul float %5, %5
  %11 = fmul float %2, 4.000000e+00
  %12 = fmul float %11, %9
  %13 = fsub float %10, %12
  %14 = fcmp olt float %13, 0.000000e+00
  br i1 %14, label %common.ret, label %next_bb

common.ret:                                       ; preds = %hit_sphere_entry, %next_bb
  %common.ret.op = phi float [ %19, %next_bb ], [ -1.000000e+00, %hit_sphere_entry ]
  ret float %common.ret.op

next_bb:                                          ; preds = %hit_sphere_entry
  %15 = fmul float %2, 2.000000e+00
  %16 = fneg float %5
  %17 = tail call float @llvm.sqrt.f32(float %13)
  %18 = fsub float %16, %17
  %19 = fdiv float %18, %15
  br label %common.ret
}

; Function Attrs: nofree nounwind memory(write, inaccessiblemem: readwrite)
define noalias noundef ptr @image(i32 noundef %width) local_unnamed_addr #3 {
image_entry:
  %0 = sitofp i32 %width to float
  %1 = fdiv float %0, 0x3FFC71C720000000
  %2 = fptosi float %1 to i32
  %3 = tail call i32 @llvm.smax.i32(i32 %2, i32 1)
  %4 = tail call ptr @_traverse_array0(i32 %width, i32 %3)
  ret ptr %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read)
define <3 x float> @pixel(ptr nocapture noundef nonnull readonly %r) local_unnamed_addr #1 {
pixel_entry:
  %deref_temp.unpack.i = load <3 x float>, ptr %r, align 16
  %deref_temp.elt4.i = getelementptr inbounds i8, ptr %r, i64 16
  %deref_temp.unpack5.i = load <3 x float>, ptr %deref_temp.elt4.i, align 16
  %0 = fsub <3 x float> <float 0.000000e+00, float 0.000000e+00, float -1.000000e+00>, %deref_temp.unpack.i
  %1 = fmul <3 x float> %deref_temp.unpack5.i, %deref_temp.unpack5.i
  %2 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %1)
  %3 = fmul <3 x float> %deref_temp.unpack5.i, %0
  %4 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %3)
  %5 = fmul float %4, -2.000000e+00
  %6 = fmul <3 x float> %0, %0
  %7 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %6)
  %8 = fadd float %7, -2.500000e-01
  %9 = fmul float %5, %5
  %10 = fmul float %2, 4.000000e+00
  %11 = fmul float %10, %8
  %12 = fsub float %9, %11
  %13 = fcmp olt float %12, 0.000000e+00
  br i1 %13, label %next_bb, label %hit_sphere.exit

hit_sphere.exit:                                  ; preds = %pixel_entry
  %14 = fmul float %2, 2.000000e+00
  %15 = fneg float %5
  %16 = tail call float @llvm.sqrt.f32(float %12)
  %17 = fsub float %15, %16
  %18 = fdiv float %17, %14
  %19 = fcmp ogt float %18, 0.000000e+00
  br i1 %19, label %then_bb, label %next_bb

common.ret:                                       ; preds = %next_bb, %then_bb
  %common.ret.op = phi <3 x float> [ %28, %then_bb ], [ %36, %next_bb ]
  ret <3 x float> %common.ret.op

then_bb:                                          ; preds = %hit_sphere.exit
  %.splatinsert.i = insertelement <3 x float> poison, float %18, i64 0
  %.splat.i = shufflevector <3 x float> %.splatinsert.i, <3 x float> poison, <3 x i32> zeroinitializer
  %20 = fmul <3 x float> %deref_temp.unpack5.i, %.splat.i
  %21 = fadd <3 x float> %deref_temp.unpack.i, %20
  %22 = fadd <3 x float> %21, <float -0.000000e+00, float -0.000000e+00, float 1.000000e+00>
  %23 = fmul <3 x float> %22, %22
  %24 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %23)
  %25 = tail call float @llvm.sqrt.f32(float %24)
  %.splatinsert = insertelement <3 x float> poison, float %25, i64 0
  %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
  %26 = fdiv <3 x float> %22, %.splat
  %27 = fadd <3 x float> %26, <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  %28 = fmul <3 x float> %27, <float 5.000000e-01, float 5.000000e-01, float 5.000000e-01>
  br label %common.ret

next_bb:                                          ; preds = %pixel_entry, %hit_sphere.exit
  %29 = tail call float @llvm.sqrt.f32(float %2)
  %30 = extractelement <3 x float> %deref_temp.unpack5.i, i64 1
  %31 = fdiv float %30, %29
  %32 = fadd float %31, 1.000000e+00
  %33 = fmul float %32, 5.000000e-01
  %34 = fsub float 1.000000e+00, %33
  %.splatinsert5 = insertelement <3 x float> poison, float %34, i64 0
  %.splat6 = shufflevector <3 x float> %.splatinsert5, <3 x float> poison, <3 x i32> zeroinitializer
  %.splatinsert7 = insertelement <3 x float> poison, float %33, i64 0
  %.splat8 = shufflevector <3 x float> %.splatinsert7, <3 x float> poison, <3 x i32> zeroinitializer
  %35 = fmul <3 x float> %.splat8, <float 5.000000e-01, float 0x3FE6666660000000, float 1.000000e+00>
  %36 = fadd <3 x float> %.splat6, %35
  br label %common.ret
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <3 x i32> @to_rgb(<3 x float> noundef %v) local_unnamed_addr #2 {
to_rgb_entry:
  %0 = fmul <3 x float> %v, <float 0x406FFFF7C0000000, float 0x406FFFF7C0000000, float 0x406FFFF7C0000000>
  %1 = fptosi <3 x float> %0 to <3 x i32>
  ret <3 x i32> %1
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define <3 x float> @unit_vector(<3 x float> noundef %v) local_unnamed_addr #2 {
unit_vector_entry:
  %0 = fmul <3 x float> %v, %v
  %1 = tail call float @llvm.vector.reduce.fadd.v3f32(float 0.000000e+00, <3 x float> %0)
  %2 = tail call float @llvm.sqrt.f32(float %1)
  %.splatinsert = insertelement <3 x float> poison, float %2, i64 0
  %.splat = shufflevector <3 x float> %.splatinsert, <3 x float> poison, <3 x i32> zeroinitializer
  %3 = fdiv <3 x float> %v, %.splat
  ret <3 x float> %3
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.vector.reduce.fadd.v3f32(float, <3 x float>) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.sqrt.f32(float) #5

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #6

attributes #0 = { nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #3 = { nofree nounwind memory(write, inaccessiblemem: readwrite) }
attributes #4 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }
attributes #5 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!0 = !{!"branch_weights", i32 1073741824, i32 0}
