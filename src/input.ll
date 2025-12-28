; ModuleID = 'input.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @barrier()

define void @test_kernel(float* %in, float* %out, i32 %id, i32 %tid, float* %shared_data) {
  ; 1. Load from global to shared
  %1 = zext i32 %id to i64
  %2 = getelementptr inbounds float, float* %in, i64 %1
  %3 = load float, float* %2, align 4
  
  %4 = zext i32 %tid to i64
  %5 = getelementptr inbounds float, float* %shared_data, i64 %4
  store float %3, float* %5, align 4

  ; 2. Sync
  call void @barrier()

  ; 3. Read NEIGHBOR's data (tid ^ 1)
  %6 = xor i32 %tid, 1
  %7 = zext i32 %6 to i64
  %8 = getelementptr inbounds float, float* %shared_data, i64 %7
  %9 = load float, float* %8, align 4
  
  ; 4. Write to output
  %10 = getelementptr inbounds float, float* %out, i64 %1
  store float %9, float* %10, align 4
  ret void
}