; ModuleID = 'shared.metal'
source_filename = "shared.metal"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"

; Function Attrs: convergent mustprogress nounwind willreturn
define void @shared_mem_kernel(float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %0, float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %1, i32 noundef %2, i32 noundef %3, float addrspace(3)* nocapture noundef "air-buffer-no-alias" %4) local_unnamed_addr #0 {
  %6 = zext i32 %2 to i64
  %7 = getelementptr inbounds float, float addrspace(1)* %0, i64 %6
  %8 = load float, float addrspace(1)* %7, align 4, !tbaa !24, !alias.scope !28, !noalias !31
  %9 = zext i32 %3 to i64
  %10 = getelementptr inbounds float, float addrspace(3)* %4, i64 %9
  store float %8, float addrspace(3)* %10, align 4, !tbaa !24, !alias.scope !34, !noalias !35
  tail call void @air.wg.barrier(i32 2, i32 1) #2
  %11 = load float, float addrspace(3)* %10, align 4, !tbaa !24, !alias.scope !34, !noalias !35
  %12 = fadd fast float %11, 1.000000e+00
  %13 = getelementptr inbounds float, float addrspace(1)* %1, i64 %6
  store float %12, float addrspace(1)* %13, align 4, !tbaa !24, !alias.scope !36, !noalias !37
  ret void
}

; Function Attrs: convergent mustprogress nounwind willreturn
declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1

attributes #0 = { convergent mustprogress nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8}
!air.kernel = !{!9}
!air.compile_options = !{!17, !18, !19}
!llvm.ident = !{!20}
!air.version = !{!21}
!air.language_version = !{!22}
!air.source_file_name = !{!23}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 7, !"frame-pointer", i32 2}
!3 = !{i32 7, !"air.max_device_buffers", i32 31}
!4 = !{i32 7, !"air.max_constant_buffers", i32 31}
!5 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
!6 = !{i32 7, !"air.max_textures", i32 128}
!7 = !{i32 7, !"air.max_read_write_textures", i32 8}
!8 = !{i32 7, !"air.max_samplers", i32 16}
!9 = !{void (float addrspace(1)*, float addrspace(1)*, i32, i32, float addrspace(3)*)* @shared_mem_kernel, !10, !11}
!10 = !{}
!11 = !{!12, !13, !14, !15, !16}
!12 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!13 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!14 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!15 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!16 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}
!17 = !{!"air.compile.denorms_disable"}
!18 = !{!"air.compile.fast_math_enable"}
!19 = !{!"air.compile.framebuffer_fetch_enable"}
!20 = !{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}
!21 = !{i32 2, i32 7, i32 0}
!22 = !{!"Metal", i32 3, i32 2, i32 0}
!23 = !{!"/Users/sueszli/dev/llvm-to-air/shared.metal"}
!24 = !{!25, !25, i64 0}
!25 = !{!"float", !26, i64 0}
!26 = !{!"omnipotent char", !27, i64 0}
!27 = !{!"Simple C++ TBAA"}
!28 = !{!29}
!29 = distinct !{!29, !30, !"air-alias-scope-arg(0)"}
!30 = distinct !{!30, !"air-alias-scopes(shared_mem_kernel)"}
!31 = !{!32, !33}
!32 = distinct !{!32, !30, !"air-alias-scope-arg(1)"}
!33 = distinct !{!33, !30, !"air-alias-scope-arg(4)"}
!34 = !{!33}
!35 = !{!29, !32}
!36 = !{!32}
!37 = !{!29, !33}
