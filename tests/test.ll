target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
target triple = "air64_v27-apple-macosx15.0.0"
; ModuleID = 'input.ll'

declare void @barrier()

define void @test_kernel(float addrspace(1)* nocapture noundef readonly "air-buffer-no-alias" %in, float addrspace(1)* nocapture noundef writeonly "air-buffer-no-alias" %out, i32 noundef %id, i32 noundef %tid, float addrspace(3)* nocapture noundef "air-buffer-no-alias" %shared_data) local_unnamed_addr #0 {
  ; 1. Load from global to shared
  %1 = zext i32 %id to i64
  %2 = getelementptr inbounds float, float addrspace(1)* %in, i64 %1
  %3 = load float, float addrspace(1)* %2, align 4
  
  %4 = zext i32 %tid to i64
  %5 = getelementptr inbounds float, float addrspace(3)* %shared_data, i64 %4
  store float %3, float addrspace(3)* %5, align 4

  ; 2. Sync
  tail call void @air.wg.barrier(i32 2, i32 1) #2

  ; 3. Read NEIGHBOR's data (tid ^ 1)
  %6 = xor i32 %tid, 1
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds float, float addrspace(3)* %shared_data, i64 %7
  %9 = load float, float addrspace(3)* %8, align 4
  
  ; 4. Write to output
  %10 = getelementptr inbounds float, float addrspace(1)* %out, i64 %1
  store float %9, float addrspace(1)* %10, align 4
  ret void
}

attributes #0 = { argmemonly mustprogress nofree norecurse nosync nounwind willreturn "approx-func-fp-math"="true" "frame-pointer"="all" "min-legal-vector-width"="0" "no-builtins" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" }

!air.kernel = !{!37}
!air.compile_options = !{!15, !16, !17}
!llvm.ident = !{!18}
!air.version = !{!19}
!air.language_version = !{!20}
!air.source_file_name = !{!21}
!15 = !{!"air.compile.denorms_disable"}
!16 = !{!"air.compile.fast_math_enable"}
!17 = !{!"air.compile.framebuffer_fetch_enable"}
!18 = !{!"Apple metal version 32023.830 (metalfe-32023.830.2)"}
!19 = !{i32 2, i32 7, i32 0}
!20 = !{!"Metal", i32 3, i32 2, i32 0}
!21 = !{!"input.ll"}
!30 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"in"}
!31 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"out"}
!32 = !{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint", !"air.arg_name", !"id"}
!33 = !{i32 3, !"air.thread_position_in_threadgroup", !"air.arg_type_name", !"uint", !"air.arg_name", !"tid"}
!34 = !{i32 4, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 4, !"air.arg_type_align_size", i32 4, !"air.arg_type_name", !"float", !"air.arg_name", !"shared_data"}
!35 = !{!30, !31, !32, !33, !34}
!36 = !{}
!37 = !{void (float addrspace(1)*, float addrspace(1)*, i32, i32, float addrspace(3)*)* @test_kernel, !36, !35}

declare void @air.wg.barrier(i32, i32) local_unnamed_addr #1
attributes #1 = { convergent mustprogress nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }
