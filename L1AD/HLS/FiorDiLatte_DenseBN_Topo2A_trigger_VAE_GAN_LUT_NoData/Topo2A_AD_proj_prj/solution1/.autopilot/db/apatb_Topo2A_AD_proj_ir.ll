; ModuleID = '/u1/hjia625/conifer/FiorDiLatte_DenseBN_Topo2A_trigger_VAE_GAN_quan_NoData/Topo2A_AD_proj_prj/solution1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

%"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>" = type { %"struct.ap_fixed_base<19, 11, true, AP_TRN, AP_WRAP, 0>" }
%"struct.ap_fixed_base<19, 11, true, AP_TRN, AP_WRAP, 0>" = type { %"struct.ssdm_int<19, true>" }
%"struct.ssdm_int<19, true>" = type { i19 }

; Function Attrs: inaccessiblemem_or_argmemonly noinline willreturn
define void @apatb_Topo2A_AD_proj_ir(%"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly "fpga.decayed.dim.hint"="44" %inputs, %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull "fpga.decayed.dim.hint"="3" "partition" %layer9_out) local_unnamed_addr #0 {
entry:
  %inputs_copy3 = alloca i836, align 512
  %layer9_out_copy_0 = alloca i19, align 512
  %layer9_out_copy_1 = alloca i19, align 512
  %layer9_out_copy_2 = alloca i19, align 512
  %0 = bitcast %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* %inputs to [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]*
  %1 = bitcast %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* %layer9_out to [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]*
  call void @copy_in([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %0, i836* nonnull align 512 %inputs_copy3, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %1, i19* nonnull align 512 %layer9_out_copy_0, i19* nonnull align 512 %layer9_out_copy_1, i19* nonnull align 512 %layer9_out_copy_2)
  call void @apatb_Topo2A_AD_proj_hw(i836* %inputs_copy3, i19* %layer9_out_copy_0, i19* %layer9_out_copy_1, i19* %layer9_out_copy_2)
  call void @copy_back([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %0, i836* %inputs_copy3, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %1, i19* %layer9_out_copy_0, i19* %layer9_out_copy_1, i19* %layer9_out_copy_2)
  ret void
}

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #1

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"(i19* nocapture "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i19* nocapture "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i19* nocapture "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %dst.addr.0.0.06.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %dst.addr.0.0.06.exit ]
  %src.addr.0.0.05 = getelementptr [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"], [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = bitcast i19* %src.addr.0.0.05 to i24*
  %2 = load i24, i24* %1
  %3 = trunc i24 %2 to i19
  switch i64 %for.loop.idx2, label %dst.addr.0.0.06.case.2 [
    i64 0, label %dst.addr.0.0.06.case.0
    i64 1, label %dst.addr.0.0.06.case.1
  ]

dst.addr.0.0.06.case.0:                           ; preds = %for.loop
  store i19 %3, i19* %dst_0, align 4
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.1:                           ; preds = %for.loop
  store i19 %3, i19* %dst_1, align 4
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.case.2:                           ; preds = %for.loop
  %4 = icmp eq i64 %for.loop.idx2, 2
  call void @llvm.assume(i1 %4)
  store i19 %3, i19* %dst_2, align 4
  br label %dst.addr.0.0.06.exit

dst.addr.0.0.06.exit:                             ; preds = %dst.addr.0.0.06.case.2, %dst.addr.0.0.06.case.1, %dst.addr.0.0.06.case.0
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %dst.addr.0.0.06.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"(i19* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.0" %dst_0, i19* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.1" %dst_1, i19* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0.2" %dst_2, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #3 {
entry:
  %0 = icmp eq [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"(i19* %dst_0, i19* %dst_1, i19* %dst_2, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 3)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.54"([3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i19* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i19* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i19* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0.2" %src_2, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %src.addr.0.0.05.exit, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %src.addr.0.0.05.exit ]
  %dst.addr.0.0.06 = getelementptr [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"], [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  switch i64 %for.loop.idx2, label %src.addr.0.0.05.case.2 [
    i64 0, label %src.addr.0.0.05.case.0
    i64 1, label %src.addr.0.0.05.case.1
  ]

src.addr.0.0.05.case.0:                           ; preds = %for.loop
  %1 = bitcast i19* %src_0 to i24*
  %2 = load i24, i24* %1
  %3 = trunc i24 %2 to i19
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.1:                           ; preds = %for.loop
  %4 = bitcast i19* %src_1 to i24*
  %5 = load i24, i24* %4
  %6 = trunc i24 %5 to i19
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.case.2:                           ; preds = %for.loop
  %7 = icmp eq i64 %for.loop.idx2, 2
  call void @llvm.assume(i1 %7)
  %8 = bitcast i19* %src_2 to i24*
  %9 = load i24, i24* %8
  %10 = trunc i24 %9 to i19
  br label %src.addr.0.0.05.exit

src.addr.0.0.05.exit:                             ; preds = %src.addr.0.0.05.case.2, %src.addr.0.0.05.case.1, %src.addr.0.0.05.case.0
  %11 = phi i19 [ %3, %src.addr.0.0.05.case.0 ], [ %6, %src.addr.0.0.05.case.1 ], [ %10, %src.addr.0.0.05.case.2 ]
  store i19 %11, i19* %dst.addr.0.0.06, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %src.addr.0.0.05.exit, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.51"([3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i19* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.0" %src_0, i19* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.1" %src_1, i19* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0.2" %src_2) #3 {
entry:
  %0 = icmp eq [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.54"([3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i19* %src_0, i19* %src_1, i19* %src_2, i64 3)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.64"(i836* nocapture "orig.arg.no"="0" "unpacked"="0.0" %dst, i64 %dst_shift, [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* readonly "orig.arg.no"="1" "unpacked"="1" %src, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %src.addr.0.0.05 = getelementptr [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"], [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %1 = mul i64 19, %for.loop.idx2
  %2 = add i64 %dst_shift, %1
  %3 = bitcast i19* %src.addr.0.0.05 to i24*
  %4 = load i24, i24* %3
  %5 = trunc i24 %4 to i19
  %6 = bitcast i836* %dst to i840*
  %7 = load i840, i840* %6
  %8 = trunc i840 %7 to i836
  %9 = zext i64 %2 to i836
  %10 = shl i836 524287, %9
  %11 = zext i19 %5 to i836
  %12 = shl i836 %11, %9
  %thr.xor1 = xor i836 %10, -1
  %thr.and2 = and i836 %8, %thr.xor1
  %thr.or3 = or i836 %thr.and2, %12
  store i836 %thr.or3, i836* %dst, align 128
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.61"(i836* noalias nocapture align 512 "orig.arg.no"="0" "unpacked"="0.0" %dst, [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="1" "unpacked"="1" %src) #3 {
entry:
  %0 = icmp eq [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %src, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.64"(i836* %dst, i64 0, [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %src, i64 44)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_in([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="0" "unpacked"="0", i836* noalias nocapture align 512 "orig.arg.no"="1" "unpacked"="1.0", [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias readonly "orig.arg.no"="2" "unpacked"="2", i19* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0.0" %_0, i19* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0.1" %_1, i19* noalias nocapture align 512 "orig.arg.no"="3" "unpacked"="3.0.2" %_2) #4 {
entry:
  call void @"onebyonecpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.61"(i836* align 512 %1, [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %0)
  call void @"onebyonecpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"(i19* align 512 %_0, i19* align 512 %_1, i19* align 512 %_2, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %2)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @"arraycpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* "orig.arg.no"="0" "unpacked"="0" %dst, i836* nocapture readonly "orig.arg.no"="1" "unpacked"="1.0" %src, i64 %src_shift, i64 "orig.arg.no"="2" "unpacked"="2" %num) #2 {
entry:
  %0 = icmp eq [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %1 = mul i64 19, %for.loop.idx2
  %2 = add i64 %src_shift, %1
  %dst.addr.0.0.06 = getelementptr [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"], [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, i64 0, i64 %for.loop.idx2, i32 0, i32 0, i32 0
  %3 = bitcast i836* %src to i840*
  %4 = load i840, i840* %3
  %5 = trunc i840 %4 to i836
  %6 = zext i64 %2 to i836
  %7 = lshr i836 %5, %6
  %8 = trunc i836 %7 to i19
  store i19 %8, i19* %dst.addr.0.0.06, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @"onebyonecpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0" %dst, i836* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0" %src) #3 {
entry:
  %0 = icmp eq [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %dst, null
  br i1 %0, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @"arraycpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* nonnull %dst, i836* %src, i64 0, i64 44)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_out([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i836* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.0" %_0, i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.1" %_1, i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.2" %_2) #5 {
entry:
  call void @"onebyonecpy_hls.p0a44struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %0, i836* align 512 %1)
  call void @"onebyonecpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.51"([3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %2, i19* align 512 %_0, i19* align 512 %_1, i19* align 512 %_2)
  ret void
}

declare void @apatb_Topo2A_AD_proj_hw(i836*, i19*, i19*, i19*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal void @copy_back([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="0" "unpacked"="0", i836* noalias nocapture readonly align 512 "orig.arg.no"="1" "unpacked"="1.0", [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* noalias "orig.arg.no"="2" "unpacked"="2", i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.0" %_0, i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.1" %_1, i19* noalias nocapture readonly align 512 "orig.arg.no"="3" "unpacked"="3.0.2" %_2) #5 {
entry:
  call void @"onebyonecpy_hls.p0a3struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>.51"([3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %2, i19* align 512 %_0, i19* align 512 %_1, i19* align 512 %_2)
  ret void
}

define void @Topo2A_AD_proj_hw_stub_wrapper(i836*, i19*, i19*, i19*) #6 {
entry:
  %4 = alloca [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]
  %5 = alloca [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]
  call void @copy_out([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %4, i836* %0, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %5, i19* %1, i19* %2, i19* %3)
  %6 = bitcast [44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %4 to %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"*
  %7 = bitcast [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %5 to %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"*
  call void @Topo2A_AD_proj_hw_stub(%"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* %6, %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* %7)
  call void @copy_in([44 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %4, i836* %0, [3 x %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"]* %5, i19* %1, i19* %2, i19* %3)
  ret void
}

declare void @Topo2A_AD_proj_hw_stub(%"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull readonly, %"struct.ap_fixed<19, 11, AP_TRN, AP_WRAP, 0>"* noalias nocapture nonnull)

attributes #0 = { inaccessiblemem_or_argmemonly noinline willreturn "fpga.wrapper.func"="wrapper" }
attributes #1 = { nounwind willreturn }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #5 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #6 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}
!datalayout.transforms.on.top = !{!5}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
!5 = !{!6, !8, !10}
!6 = !{!7}
!7 = !{!"1.0", [3 x i19]* null}
!8 = !{!9}
!9 = !{!"array_partition", !"type=Complete", !"dim=1"}
!10 = !{!11, !12, !13}
!11 = !{!"1.0.0", i19* null}
!12 = !{!"1.0.1", i19* null}
!13 = !{!"1.0.2", i19* null}
