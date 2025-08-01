#include "AsyncUtility.h"
#include "AtomicRMWOpsEmitter.h"
#include "BufferOpsEmitter.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
using namespace mlir::triton::gpu;

using ::mlir::LLVM::getSharedMemoryBase;
using ::mlir::LLVM::AMD::getVectorSize;
using ::mlir::LLVM::AMD::llLoad;
using ::mlir::LLVM::AMD::llStore;
using ::mlir::triton::AMD::ISAFamily;
using ::mlir::triton::gpu::getTotalElemsPerThread;

namespace {

std::optional<const char *> getAMDGPUMemScopeStr(MemSyncScope scope) {
  switch (scope) {
  case MemSyncScope::GPU:
    return "agent";
  case MemSyncScope::CTA:
    return "workgroup";
  // The default AMDHSA LLVM Sync Scope is "system", so no string is
  // provided here
  case MemSyncScope::SYSTEM:
  default:
    return "";
  }
}

std::pair<bool, bool> getOrderingFlags(MemSemantic memOrdering) {
  bool emitReleaseFence = false;
  bool emitAcquireFence = false;
  switch (memOrdering) {
  case MemSemantic::RELAXED:
    // In this case, no memory fences are needed
    break;
  case MemSemantic::RELEASE:
    emitReleaseFence = true;
    break;
  case MemSemantic::ACQUIRE:
    emitAcquireFence = true;
    break;
  case MemSemantic::ACQUIRE_RELEASE:
    emitAcquireFence = true;
    emitReleaseFence = true;
  default:
    // default == acq_rel, so we emit the same barriers
    emitAcquireFence = true;
    emitReleaseFence = true;
  }
  return {emitAcquireFence, emitReleaseFence};
}

LogicalResult emitFence(Operation *op, ConversionPatternRewriter &rewriter,
                        Location loc, MemSemantic memOrdering,
                        MemSyncScope memScope, bool preAtomic) {
  // This function emits an LLVM::FenceOp which will get lowered by the
  // LLVM backend to the right scope and ordering instructions, as
  // described in the "atomicrmw" entries for "global" address-space,
  // in the "AMDHSA Memory Model Code Sequences GFX942"
  // table in https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
  //
  // Triton supports three scopes for atomic access
  // 1. System
  // 2. GPU (default) ('Agent' for AMDGPU)
  // 3. CTA ('Workgroup' for AMDGPU)
  //
  // and 4 orderings
  // 1. Relaxed
  // 2. Acquire
  // 3. Release
  // 4. AcquireRelease
  //
  // The following table shows the scope and ordering instructions that
  // are emitted by this function for each combination of scope and ordering
  // for buffer-atomic instructions.
  //
  // Note: In the following comments, "[buffer-atomic_0.. buffer-atomic_n]"
  // represents a sequence of buffer-atomic instructions that are lowered from
  // a single tl.atomic_*
  //
  // Unordered(Relaxed):
  //   agent/workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //                    No scope/ordering instrs are required.
  //   system: //TODO:
  // Acquire:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2.
  //              No scope/ordering instrs are required.
  //   agent: Instr seq: [buffer-atomic_0.. buffer-atomic_n],
  //                     s_waitcnt vmcnt(0), buffer_inv sc1=1
  //          Waves across an agent may use different L1 and L2.
  //          Atomic ops bypass L1 and operate on L2.
  //          s_waitcnt vmcnt(0) ensures that the atomicrmw has completed
  //          before invalidating the cache. buffer_inv sc1=1 will a) L1:
  //          invalidate cache b) L2: Invalidate non-coherently modified lines
  //          if multiple L2s are configured, NOP otherwise. This buffer_inv
  //          ensures that following loads do not see stale global values.
  //   system: //TODO:
  //
  // Release:
  //   workgroup: Instr seq: [buffer-atomic_0.. buffer-atomic_n]
  //              All waves in the workgroup use same L1 and L2 so all
  //              previous global writes of a waver are visible to all other
  //              waves in the workgroup. LDS operations for all waves are
  //              executed in a total global ordering and are observed by all
  //              waves in the workgroup. So LDS stores issued before the
  //              release will be visible to LDS loads after the read of the
  //              released buffer-atomic. So, swait_cnt lgkmcnt is not
  //              required.
  //   agent: Instr seq: buffer_wbl2 sc1=1, s_waitcnt vmcnt(0),
  //                     [buffer-atomic_0.. buffer-atomic_n]
  //          buffer_wbl2 sc1=1 ensures that dirtly L2 lines are visible to
  //          CUs that don't use the same L2.
  //          From SIMemoryLegalizer.cpp SIGfx940CacheControl::insertRelease:
  //            "Inserting a "S_WAITCNT vmcnt(0)" before is not required
  //             because the hardware does not reorder memory operations by
  //             the same wave with respect to a following "BUFFER_WBL2".
  //             The "BUFFER_WBL2" is guaranteed to initiate writeback of
  //             any dirty cache lines of earlier writes by the same wave.
  //             A "S_WAITCNT vmcnt(0)" is needed after to ensure the writeback
  //             has completed.""
  //   system: //TODO:
  //
  // AcquireRelease:
  //   Instr seq: Release scope/order insts,
  //              [buffer-atomic_0..buffer-atomic_n],
  //              Acquire scope/order instrs.
  //
  // LLVM::FenceOp lowering will emit the required cache ops and s_waitcnt
  // vmcnt(0) instrs

  auto [emitReleaseFence, emitAcquireFence] = getOrderingFlags(memOrdering);
  if (MemSyncScope::SYSTEM == memScope)
    return rewriter.notifyMatchFailure(
        op, "System memory scope is not supported for Buffer Atomic Ops");
  auto scopeStr = getAMDGPUMemScopeStr(memScope);
  if (!scopeStr)
    return rewriter.notifyMatchFailure(
        op, "Unsupported memory scope for Buffer Atomic Ops");

  StringAttr scope = mlir::StringAttr::get(loc.getContext(), *scopeStr);

  if (emitReleaseFence && preAtomic) {
    rewriter.create<LLVM::FenceOp>(loc, TypeRange{},
                                   LLVM::AtomicOrdering::release, scope);
  }

  if (emitAcquireFence && !preAtomic) {
    rewriter.create<LLVM::FenceOp>(loc, TypeRange{},
                                   LLVM::AtomicOrdering::acquire, scope);
  }
  return success();
}

// Return a predicate that is true only if the current thread holds unique data,
// according to freeVarsMask.
Value emitRedundantThreadPredicate(
    const llvm::MapVector<StringAttr, int32_t> &freeVarMasks,
    ConversionPatternRewriter &rewriter, Location loc,
    const AMD::TargetInfo &targetInfo) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ctx = rewriter.getContext();
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");

  Value zero = b.i32_val(0);
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId = freeVarMasks.lookup(kBlock) == 0
                      ? zero
                      : targetInfo.getClusterCTAId(rewriter, loc);

  Value pred = b.true_val();
  auto dimNames = {kLane, kWarp, kBlock};
  auto dimIds = {laneId, warpId, blockId};
  for (auto [dimName, dimId] : llvm::zip(dimNames, dimIds)) {
    int32_t mask = freeVarMasks.lookup(dimName);
    if (mask != 0) {
      auto dimPred = b.icmp_eq(b.and_(dimId, b.i32_val(mask)), zero);
      pred = b.and_(pred, dimPred);
    }
  }
  return pred;
}

// Contains some helper functions for both Load and Store conversions.
struct LoadStoreConversionBase {
  explicit LoadStoreConversionBase(const AMD::TargetInfo &targetInfo,
                                   ModuleAxisInfoAnalysis &axisAnalysisPass)
      : targetInfo(targetInfo), axisAnalysisPass(axisAnalysisPass) {}

  // Create a LLVM vector of type `vecTy` containing all zeros
  Value createZeroVector(OpBuilder &builder, Location loc,
                         VectorType vecTy) const {
    mlir::Attribute zeroAttr = builder.getZeroAttr(vecTy.getElementType());
    auto denseValue =
        DenseElementsAttr::get(cast<mlir::ShapedType>(vecTy), zeroAttr);
    Value zeroVal = builder.create<LLVM::ConstantOp>(loc, vecTy, denseValue);
    return zeroVal;
  }

  // Given a vector of values `elems` and a starting point `start`, create a
  // LLVM vector of length `vec` whose elements are `elems[start, ...,
  // elems+vec-1]`
  Value packElementRangeIntoVector(ConversionPatternRewriter &rewriter,
                                   const LLVMTypeConverter *typeConverter,
                                   Location loc, VectorType vecTy,
                                   ArrayRef<Value> elems, int64_t start) const {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    int64_t vec = vecTy.getNumElements();
    // If we need to mask the loaded value with other elements
    Value v = b.undef(vecTy);
    for (size_t s = 0; s < vec; ++s) {
      Value otherElem = elems[start + s];
      Value indexVal =
          LLVM::createIndexConstant(rewriter, loc, typeConverter, s);
      v = b.insert_element(vecTy, v, otherElem, indexVal);
    }
    return v;
  }

  // Return a tensor of pointers with the same type of `basePtr` and the same
  // shape of `offset`
  Type getPointerTypeWithShape(Value basePtr, Value offset) const {
    Type basePtrType = basePtr.getType();
    auto offsetType = cast<RankedTensorType>(offset.getType());
    return offsetType.cloneWith(std::nullopt, basePtrType);
  }

  // Unpack the elements contained in a `llvmStruct` into a `SmallVector` of
  // `Value`s. While you do that, check also the alignment of the mask and
  // update the vector length `vec` accordingly
  SmallVector<Value>
  getMaskElemsAndUpdateVeclen(ConversionPatternRewriter &rewriter, Location loc,
                              Value llMask, Value mask, unsigned &vec) const {
    SmallVector<Value> maskElems;
    if (llMask) {
      vec = std::min<size_t>(vec, getMaskAlignment(mask));
      maskElems = unpackLLElements(loc, llMask, rewriter);
    }
    return maskElems;
  }

  unsigned getMaskAlignment(Value mask) const {
    return axisAnalysisPass.getMaskAlignment(mask);
  }

protected:
  const AMD::TargetInfo &targetInfo;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Contains some helper functions for direct to lds loads.
struct DirectToLdsLoadConversionBase : public LoadStoreConversionBase {
  explicit DirectToLdsLoadConversionBase(
      const AMD::TargetInfo &targetInfo,
      ModuleAxisInfoAnalysis &axisAnalysisPass)
      : LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  // direct to lds loads do not support per lane shared offsets. We need to
  // ensure that we write coalesced into shared memory. This means we cannot
  // exceed the supported load width because splitting them would cause strided
  // (non coalesced) writes. Additionally:
  //   1) For *non* swizzled shared encodings we check if they result in
  //      coalesced writes and can then lower them directly to the intrinsics.
  //   2) For swizzled shared encodings we need to transfer the swizzling to the
  //      source pointers. For now this is done by swizzling the pointers
  //      between the lane of a warp via permute. This only works if the swizzle
  //      pattern does not exchange elements between warps which holds for all
  //      our swizzle patterns. There is still a check performed to not silently
  //      produce wrong results if we invalidate the condition in the future
  LogicalResult canWriteCoalesced(RewriterBase &rewriter, Operation *op,
                                  RankedTensorType srcTy, MemDescType dstTy,
                                  unsigned vectorSize,
                                  bool hasSwizzling) const {
    int vecBits = vectorSize * dstTy.getElementTypeBitWidth();
    if (!targetInfo.supportsDirectToLdsLoadBitWidth(vecBits)) {
      LDBG(op << " results in unsupported load bitwidth: " << vecBits);
      return failure();
    }
    // Compute the blocked -> shared linear layout to check preconditions
    LinearLayout srcLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout = triton::gpu::toLinearLayout(dstTy);
    LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

    unsigned threadsPerWarp = lookupThreadsPerWarp(rewriter);
    if (!hasSwizzling && !LLVM::AMD::canCoalesceWriteIntoSharedMemory(
                             rewriter, srcToSharedLayout, threadsPerWarp)) {
      LDBG(op << " does not write coalesced into LDS and is not swizzled");
      return failure();
    }

    if (hasSwizzling && !LLVM::AMD::doesSwizzleInsideWarp(
                            rewriter, srcToSharedLayout, threadsPerWarp)) {
      LDBG(op << " does swizzle across warp boundaries");
      return failure();
    }
    return success();
  }

  // Determine the vecTy per async load to LDS and collect the warp uniform
  // shared memory addresses per vecTy. This will only emit the address
  // calculation and not the actual loads.
  std::pair<SmallVector<Value>, VectorType> emitWarpStartSharedAddresses(
      RewriterBase &rewriter, Operation *op, RankedTensorType srcTy,
      MemDescType dstTy, bool hasSwizzling, Type resElemTy, Value llDst) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // On GFX9 the shared memory address is a scalar so we need to compute the
    // start address by setting lane_id to 0 and ignore swizzling

    if (hasSwizzling) {
      // Overwrite the shared encoding with a non swizzled
      // one to get the base address of the warp
      // TODO (alex): this is only correct as long as the lds view is a
      // contigous block. So this can break if we slice along the 2 minor
      // dimensions.
      auto dstEnc = cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
      auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
          op->getContext(), dstEnc.getVec(), 1, 1, dstEnc.getOrder(),
          dstEnc.getCTALayout());
      dstTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                               flatSharedEnc, dstTy.getMemorySpace());
    }

    auto smemObj = mlir::LLVM::getSharedMemoryObjectFromStruct(
        loc, llDst, resElemTy, rewriter);
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    // The warp ID is always a scalar so we use the ThreadId from the first lane
    // to compute the warp ID which improves codegen. shuffleIdx will be lowered
    // to readlane 0 placing it into a SGPR and hinting at LLVM that it
    // should use scalar ops which also allows it to better hoist values
    Value tid =
        targetInfo.shuffleIdx(rewriter, loc, getThreadId(rewriter, loc), 0);
    int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
    Value warpSizeVal = b.i32_val(threadsPerWarp);
    Value warpId = b.udiv(tid, warpSizeVal);

    VectorType vecTy;
    SmallVector<Value> warpStartAddrs;
    bool ok = emitTransferBetweenRegistersAndShared(
        regLayout, dstTy, resElemTy, {}, smemObj, loc, rewriter, targetInfo,
        /*laneId=*/b.i32_val(0), warpId,
        [&](VectorType vecTy_, Value shmemAddr) {
          vecTy = vecTy_;
          warpStartAddrs.push_back(shmemAddr);
        });
    assert(ok);

    return {std::move(warpStartAddrs), vecTy};
  }

  // For each load emit the computation to get the lane id offset which holds
  // the source pointers/offsets we need to store to shared memory
  SmallVector<Value>
  emitSwizzledLaneOffsets(RewriterBase &rewriter, Operation *op,
                          RankedTensorType srcTy, MemDescType dstTy,
                          bool hasSwizzling, Value llDst, Type resElemTy,
                          VectorType vecTy) const {
    auto loc = op->getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Create the non swizzled/flat encoding
    auto dstEnc = cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    auto flatSharedEnc = SwizzledSharedEncodingAttr::get(
        srcTy.getContext(), dstEnc.getVec(), 1, 1, dstEnc.getOrder(),
        dstEnc.getCTALayout());
    auto flatTy = MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                   flatSharedEnc, dstTy.getMemorySpace());

    // Create regToShared layout for the swizzled and flat encoding
    auto regLayout = triton::gpu::toLinearLayout(srcTy);

    auto sharedSwizz = triton::gpu::toLinearLayout(dstTy);
    auto sharedFlat = triton::gpu::toLinearLayout(flatTy);

    auto regToSharedSwizzled = regLayout.invertAndCompose(sharedSwizz);
    auto regToSharedFlat = regLayout.invertAndCompose(sharedFlat);

    MLIRContext *ctx = rewriter.getContext();
    StringAttr kBlock = str_attr("block");
    StringAttr kRegister = str_attr("register");
    StringAttr kLane = str_attr("lane");
    StringAttr kWarp = str_attr("warp");
    auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
    Value blockId = b.i32_val(0);

    int numberOfLoads =
        regToSharedSwizzled.getInDimSize(kRegister) / vecTy.getNumElements();

    // For each load compute the difference between the flat and the swizzled
    // linear offsets into shared memory
    // TODO (alex): this is only correct as long as the lds view is a contigous
    // block. So this can break if we slice along the 2 minor dimensions
    SmallVector<Value> swizzledOffsets;
    swizzledOffsets.reserve(numberOfLoads);
    for (int i = 0; i < numberOfLoads; i++) {
      auto regId = b.i32_val(i * vecTy.getNumElements());

      std::array<std::pair<StringAttr, Value>, 4> indices{{
          {kRegister, regId},
          {kLane, laneId},
          {kWarp, warpId},
          {kBlock, blockId},
      }};

      Value swizzledOffset =
          applyLinearLayout(loc, rewriter, regToSharedSwizzled, indices)[0]
              .second;
      Value flatOffset =
          applyLinearLayout(loc, rewriter, regToSharedFlat, indices)[0].second;

      // Normalize the offset by vecTy to obtain the offset in lanes
      auto laneOffet = b.sdiv(b.sub(swizzledOffset, flatOffset),
                              b.i32_val(vecTy.getNumElements()));
      swizzledOffsets.push_back(laneOffet);
    }
    return swizzledOffsets;
  }

  // Swizzle the mask (1bit) based on selectLane via ballot
  Value shuffleMask(RewriterBase &rewriter, TritonLLVMOpBuilder &b,
                    Location loc, const TargetInfoBase &targetInfo,
                    Value selectLane, Value mask) const {
    auto warpMask =
        targetInfo.ballot(rewriter, loc, rewriter.getI64Type(), mask);
    // Extract the selectLane bit
    auto bitMask = b.lshr(warpMask, b.zext(rewriter.getI64Type(), selectLane));
    return b.trunc(i1_ty, bitMask);
  }

  // For direct-to-lds the order of the shared encoding decides the order we
  // load elements from global memory. This function returns true if the fastest
  // dim for the sharedEnc is contiguous for the global ptrs/offsets
  bool isFastedLoadDimContiguous(Value srcPtrOrOffset,
                                 MemDescType sharedTy) const {
    auto fastestDim = triton::gpu::getOrder(sharedTy)[0];
    AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(srcPtrOrOffset);

    // This can happen if axis analysis fails (e.g. lit tests).
    if (axisInfo->getRank() <= fastestDim)
      return false;

    return axisInfo->getContiguity(fastestDim) > 1;
  }
};

struct LoadOpConversion : public ConvertOpToLLVMPattern<triton::LoadOp>,
                          public LoadStoreConversionBase {
  LoadOpConversion(LLVMTypeConverter &converter,
                   const AMD::TargetInfo &targetInfo,
                   ModuleAxisInfoAnalysis &axisAnalysisPass,
                   PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // original values
    Value ptr = op.getPtr();
    Value mask = op.getMask();
    Value other = op.getOther();

    // adaptor values
    assert(!isTensorPointerType(ptr.getType()) &&
           "Cannot convert load with a tensor pointer into LLVM; "
           "this case should be transformed to normal load before lowering");
    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned numElems = getTotalElemsPerThread(ptr.getType());

    // Get the LLVM values for pointers
    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    assert(ptrElems.size() == numElems);

    // Get the LLVM values for mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> otherElems;
    if (other)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // vectorized iteration through all the pointer/mask/other elements
    const int valueElemNBits =
        std::max(8u, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;
    const int numVecs = numElems / vec;

    auto cacheMod = op.getCache();
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      const size_t movWidth = width < 16 ? 16 : width;
      assert(wordNElems * nWords * numVecs == numElems);

      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value ptr = ptrElems[vecStart];

      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // If we need to mask the loaded value with other elements
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);

      Value loadVal =
          llLoad(rewriter, loc, ptr, vecTy, pred, falseVal, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadOp>,
      public LoadStoreConversionBase {
  BufferLoadOpConversion(LLVMTypeConverter &converter,
                         const AMD::TargetInfo &targetInfo,
                         ModuleAxisInfoAnalysis &axisAnalysisPass,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value other = op.getOther();
    auto cacheMod = op.getCache();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = op.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);
    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offset
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    assert(offsetElems.size() == numElems);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    // Get the `other` value (if any)
    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    // Create the resource descriptor and then emit the buffer_load intrinsic(s)
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;
    Type vecTy = LLVM::getVectorType(valueElemTy, vec);
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Value pred = mask ? maskElems[vecStart] : b.int_val(1, 1);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      if (otherElems.size() != 0)
        falseVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
            otherElems, vecStart);
      Value loadVal = bufferEmitter.emitLoad(
          vecTy, rsrcDesc, offsetElems[vecStart], pred, falseVal, cacheMod);
      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    Type llvmResultStructTy = getTypeConverter()->convertType(valueTy);
    Value resultStruct = packLLElements(loc, getTypeConverter(), loadedVals,
                                        rewriter, llvmResultStructTy);

    rewriter.replaceOp(op, {resultStruct});
    return success();
  }
};

struct BufferLoadToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferLoadToLocalOp>,
      public DirectToLdsLoadConversionBase {
  BufferLoadToLocalOpConversion(LLVMTypeConverter &converter,
                                const AMD::TargetInfo &targetInfo,
                                ModuleAxisInfoAnalysis &axisAnalysisPass,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferLoadToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // Original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();

    // Converted values
    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llDst = adaptor.getDest();
    Value llMask = adaptor.getMask();
    Value llOther = adaptor.getOther();
    Value llStride = adaptor.getStride();

    RankedTensorType ptrType =
        cast<RankedTensorType>(getPointerTypeWithShape(ptr, offset));
    unsigned numElems = getTotalElemsPerThread(ptrType);

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    assert(offsetElems.size() == numElems);

    SmallVector<Value> otherElems;
    if (llOther)
      otherElems = unpackLLElements(loc, llOther, rewriter);

    auto dstTy = op.getDest().getType();
    auto sharedEnc = cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    bool hasSwizzling = sharedEnc.getMaxPhase() != 1;
    if (failed(canWriteCoalesced(rewriter, op, ptrType, dstTy, vec,
                                 hasSwizzling))) {
      return failure();
    }

    auto [ldsWarpStartAddrs, vecTy] = emitWarpStartSharedAddresses(
        rewriter, op, ptrType, dstTy, hasSwizzling, resElemTy, llDst);
    assert(vecTy.getNumElements() == vec);

    SmallVector<Value> swizzledLaneOffsets;
    if (hasSwizzling) {
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, ptrType, dstTy, hasSwizzling, llDst, resElemTy, vecTy);
      assert(ldsWarpStartAddrs.size() == swizzledLaneOffsets.size());
    }

    Value laneId = getLaneId(rewriter, loc);
    int vecBytes =
        (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
    assert(llvm::isPowerOf2_32(vecBytes));
    Value vecBytesVal = b.i32_val(vecBytes);

    // Create the resource descriptor and then emit the buffer_loads to lds
    // based on the collected shared addresses and vector size
    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);

    for (int i = 0; i < ldsWarpStartAddrs.size(); i++) {
      auto srcIdx = i * vec;
      auto offsetIn = offsetElems[srcIdx];
      Value pred = mask ? maskElems[srcIdx] : b.true_val();

      if (hasSwizzling) {
        // Apply swizzling to the src offsets
        auto swizzleLaneOffset = swizzledLaneOffsets[i];
        // laneId + swizzleOffset will always stay inside the warp [0,
        // threadsPerWarp) because we only swizzle inside a warp
        Value swizzledLaneId = b.add(laneId, swizzleLaneOffset);
        if (isFastedLoadDimContiguous(offset, cast<MemDescType>(dstTy))) {
          // Because rows are contiguous and we only swizzle inside rows by
          // swapping elements between lanes we can add laneOffset * vecSize to
          // the offset to apply the swizzling
          offsetIn = b.add(offsetIn, b.mul(swizzleLaneOffset,
                                           b.i32_val(vecTy.getNumElements())));
        } else {
          // If rows are not contiguous in memory we need to shuffle the
          // pointers to apply the swizzling to the src pointers
          offsetIn =
              targetInfo.shuffleIdx(rewriter, loc, offsetIn, swizzledLaneId);
        }

        if (mask) {
          pred =
              shuffleMask(rewriter, b, loc, targetInfo, swizzledLaneId, pred);
        }
      }

      auto bufferLoadToLds = bufferEmitter.emitLoadToLds(
          vecTy, vecBytesVal, rsrcDesc, offsetIn, ldsWarpStartAddrs[i], pred,
          op.getCache());
      AMD::addAsyncCopyAliasScope(bufferLoadToLds);
      if (!otherElems.empty()) {
        Value storeVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, vecTy, otherElems, srcIdx);
        Type ptrTy = ldsWarpStartAddrs[i].getType();

        // otherDstPtr = warpStartAddr + (laneId + [swizzleLaneOffset]) * vecTy
        Value ldsAddr = b.gep(ptrTy, vecTy, ldsWarpStartAddrs[i], laneId);
        if (hasSwizzling)
          ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzledLaneOffsets[i]);

        llStore(rewriter, loc, ldsAddr, storeVal,
                b.icmp_ne(maskElems[srcIdx], b.true_val()), op.getCache(),
                /*forceNoAliasAsyncLoads=*/true);
      }
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct AsyncCopyGlobalToLocalOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::AsyncCopyGlobalToLocalOp>,
      public DirectToLdsLoadConversionBase {
  AsyncCopyGlobalToLocalOpConversion(LLVMTypeConverter &converter,
                                     const AMD::TargetInfo &targetInfo,
                                     ModuleAxisInfoAnalysis &axisAnalysisPass,
                                     PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        DirectToLdsLoadConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::gpu::AsyncCopyGlobalToLocalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto srcTy = op.getSrc().getType();
    auto srcEncoding = srcTy.getEncoding();

    if (!isa<BlockedEncodingAttr, SliceEncodingAttr>(srcEncoding))
      return rewriter.notifyMatchFailure(
          op, "requires Blocked or Slice encoding for src");
    if (srcTy.getShape().size() != 2)
      return rewriter.notifyMatchFailure(op, "only supports 2d tensors");

    auto dstTy = op.getResult().getType();
    auto sharedEnc = cast<SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    auto resElemTy = getTypeConverter()->convertType(dstTy.getElementType());

    Value llSrc = adaptor.getSrc();
    auto srcElems = unpackLLElements(loc, llSrc, rewriter);
    Value llDst = adaptor.getResult();

    // We can load N elements at a time if:
    //  1. Every group of N source pointers are contiguous.  For example, if
    //     N=2, then the pointers should be [x, x+1, y, y+1, ...].
    //  2. The mask (if present) has "alignment" N, meaning that each group of N
    //     mask bits are the same.  For example if N=2, the mask must be
    //     [x, x, y, y, ...].
    unsigned maxVec = getVectorSize(op.getSrc(), axisAnalysisPass);
    auto maskElements = getMaskElemsAndUpdateVeclen(
        rewriter, loc, adaptor.getMask(), op.getMask(), maxVec);

    bool hasSwizzling = sharedEnc.getMaxPhase() != 1;
    if (failed(canWriteCoalesced(rewriter, op, srcTy, dstTy, maxVec,
                                 hasSwizzling))) {
      return failure();
    }

    Value llMask = adaptor.getMask();
    SmallVector<Value> maskElems;
    if (llMask) {
      maskElems = unpackLLElements(loc, llMask, rewriter);
      assert(srcElems.size() == maskElems.size());
    }

    SmallVector<Value> otherElems;
    if (op.getOther()) {
      otherElems = unpackLLElements(loc, adaptor.getOther(), rewriter);
      assert(srcElems.size() == otherElems.size());
    }

    auto [ldsWarpStartAddrs, vecTy] = emitWarpStartSharedAddresses(
        rewriter, op, srcTy, dstTy, hasSwizzling, resElemTy, llDst);
    assert(vecTy.getNumElements() == maxVec);

    SmallVector<Value> swizzledLaneOffsets;
    if (hasSwizzling) {
      swizzledLaneOffsets = emitSwizzledLaneOffsets(
          rewriter, op, srcTy, dstTy, hasSwizzling, llDst, resElemTy, vecTy);
      assert(ldsWarpStartAddrs.size() == swizzledLaneOffsets.size());
    }

    Value laneId = getLaneId(rewriter, loc);
    int vecBytes =
        (vecTy.getNumElements() * vecTy.getElementTypeBitWidth()) / 8;
    assert(llvm::isPowerOf2_32(vecBytes));
    Value vecBytesVal = b.i32_val(vecBytes);
    int32_t cacheModifiers =
        mlir::LLVM::AMD::getCtrlBitsForCacheModifierOnTarget(
            op.getCache(), /*isLoad=*/true, targetInfo);

    // Emit the load to lds based on the collected shared addresses and vector
    // size
    for (int i = 0; i < ldsWarpStartAddrs.size(); i++) {
      auto srcIdx = i * maxVec;
      Value srcPtr = srcElems[srcIdx];
      Value pred = maskElements.empty() ? b.true_val() : maskElems[srcIdx];

      // Apply swizzling to the src pointers
      if (hasSwizzling) {
        // laneId + laneOffset will always stay inside the warp [0,
        // threadsPerWarp) because we only swizzle inside a warp
        Value swizzledLaneId =
            b.add(getLaneId(rewriter, loc), swizzledLaneOffsets[i]);

        if (isFastedLoadDimContiguous(op.getSrc(), cast<MemDescType>(dstTy))) {
          // Because rows are contiguous and we only swizzle inside rows by
          // swapping elements between lanes we can move the vecTy typed src
          // pointer by laneOffset elements to apply the swizzling.
          srcPtr =
              b.gep(srcPtr.getType(), vecTy, srcPtr, swizzledLaneOffsets[i]);
        } else {
          // If rows are not contiguous in memory we need to shuffle the
          // pointers to apply the swizzling to the src pointers
          srcPtr = targetInfo.shuffleIdx(rewriter, loc, srcPtr, swizzledLaneId);
        }
        if (!maskElements.empty()) {
          pred =
              shuffleMask(rewriter, b, loc, targetInfo, swizzledLaneId, pred);
        }
      }

      if (maskElems.empty()) {
        auto globalLoadLdsOp = rewriter.create<ROCDL::GlobalLoadLDSOp>(
            loc,
            /*globalPtr=*/srcPtr, /*ldsPtr=*/ldsWarpStartAddrs[i],
            /*size=*/vecBytes, /*offset=*/0,
            /*aux=*/cacheModifiers, /*alias_scopes=*/nullptr,
            /*noalias_scopes=*/nullptr, /*tbaa=*/nullptr);
        AMD::addAsyncCopyAliasScope(globalLoadLdsOp);
        continue;
      }

      Block *currentBlock = rewriter.getInsertionBlock();
      Block *afterLoad =
          rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
      Block *loadBlock = rewriter.createBlock(afterLoad);
      rewriter.setInsertionPointToEnd(currentBlock);
      rewriter.create<LLVM::CondBrOp>(loc, pred, loadBlock, afterLoad);
      rewriter.setInsertionPointToStart(loadBlock);
      auto globalLoadLdsOp = rewriter.create<ROCDL::GlobalLoadLDSOp>(
          loc, srcPtr, ldsWarpStartAddrs[i], vecBytes,
          /*offset=*/0, cacheModifiers, nullptr, nullptr, nullptr);
      AMD::addAsyncCopyAliasScope(globalLoadLdsOp);

      rewriter.create<LLVM::BrOp>(loc, afterLoad);
      rewriter.setInsertionPointToStart(afterLoad);
      if (!otherElems.empty()) {
        Value storeVal = packElementRangeIntoVector(
            rewriter, this->getTypeConverter(), loc, vecTy, otherElems, srcIdx);
        Type ptrTy = ldsWarpStartAddrs[i].getType();

        // otherDstPtr = warpStartAddr + (laneId + [swizzleLaneOffset]) * vecTy
        Value ldsAddr = b.gep(ptrTy, vecTy, ldsWarpStartAddrs[i], laneId);
        if (hasSwizzling)
          ldsAddr = b.gep(ptrTy, vecTy, ldsAddr, swizzledLaneOffsets[i]);

        llStore(rewriter, loc, ldsAddr, storeVal,
                b.icmp_ne(maskElems[srcIdx], b.true_val()), op.getCache(),
                /*forceNoAliasAsyncLoads=*/true);
      }
    }

    // Drop the result token.
    Value zero = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), IntegerType::get(op.getContext(), 32),
        rewriter.getI32IntegerAttr(0));
    rewriter.replaceOp(op, zero);
    return success();
  }
};

struct StoreOpConversion : public ConvertOpToLLVMPattern<triton::StoreOp>,
                           public LoadStoreConversionBase {
  StoreOpConversion(LLVMTypeConverter &converter,
                    const AMD::TargetInfo &targetInfo,
                    ModuleAxisInfoAnalysis &axisAnalysisPass,
                    PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value value = op.getValue();
    Value mask = op.getMask();

    Value llPtr = adaptor.getPtr();
    Value llMask = adaptor.getMask();
    Value llValue = adaptor.getValue();

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto valueTy = value.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));

    // Determine the vectorization size
    unsigned vec = getVectorSize(ptr, axisAnalysisPass);
    unsigned elemsPerThread = getTotalElemsPerThread(ptr.getType());

    auto ptrElems = unpackLLElements(loc, llPtr, rewriter);
    auto valueElems = unpackLLElements(loc, llValue, rewriter);
    assert(ptrElems.size() == valueElems.size());

    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    const size_t valueElemNBits =
        std::max<int>(8, valueElemTy.getIntOrFloatBitWidth());
    const size_t valueElemNBytes = valueElemNBits / 8;

    auto cacheMod = op.getCache();
    const int numVecs = elemsPerThread / vec;
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < elemsPerThread; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      auto vecTy = LLVM::getVectorType(valueElemTy, vec);

      const size_t maxWordWidth = std::max<size_t>(32, valueElemNBits);
      const size_t totalWidth = valueElemNBits * vec;
      const size_t width = std::min(totalWidth, maxWordWidth);
      const size_t nWords = std::max<size_t>(1, totalWidth / width);
      const size_t wordNElems = width / valueElemNBits;
      assert(wordNElems * nWords * numVecs == elemsPerThread);

      SmallVector<std::pair<Value, std::string>> asmArgs;
      Value elem = valueElems[vecStart];
      Value ptr = ptrElems[vecStart];

      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      llStore(rewriter, loc, ptr, storeVal, pred, cacheMod);
    } // end vec
    rewriter.eraseOp(op);
    return success();
  }
};

struct BufferAtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicRMWOp>,
      public LoadStoreConversionBase {
  BufferAtomicRMWOpConversion(LLVMTypeConverter &converter,
                              const AMD::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferAtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // v4f16 and v4bf16 variants of buffer atomics do not exist.
    // only v2f16 and v2bf16.
    if (valueElemTy.isBF16() || valueElemTy.isF16()) {
      // We clamp to the only supported vectorization width here (2).
      // In ConvertToBufferOps we check that we have a large enough vector size
      assert(vec >= 2);
      vec = 2u;
      // The max width of a buffer atomic op is 64-bits
      // Some types like F32 don't have a 2x vectorized version
    } else if (valueElemTy.isF32() || valueElemTy.isF64() ||
               valueElemTy.isInteger(32) || valueElemTy.isInteger(64)) {
      vec = 1u;
    }

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops) see: https://llvm.org/docs/AMDGPUUsage.html#memory-model-gfx942
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastRMWOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    //    We set GLC=1, to return the old value. Atomics in GFX942 execute with
    //    either device (default) or system scope (controlled by the sc1 flag).
    //    This is distinct from the memory scope of the atomic (i.e, the memory
    //    fences which appear before/after the ops).

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = std::distance(opUsers.begin(), opUsers.end()) > 0;
    auto moduleOp = op->getParentOfType<ModuleOp>();

    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value falseVal = createZeroVector(rewriter, loc, cast<VectorType>(vecTy));
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);

      Value loadVal = bufferEmitter.emitAtomicRMW(
          atomicRmwAttr, vecTy, rsrcDesc, offsetElems[vecStart], storeVal, pred,
          hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastRMWOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Acquire Fence post-atomic
    if (failed(emitFence(op, rewriter, lastRMWOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferAtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferAtomicCASOp>,
      public LoadStoreConversionBase {
  BufferAtomicCASOpConversion(LLVMTypeConverter &converter,
                              const AMD::TargetInfo &targetInfo,
                              ModuleAxisInfoAnalysis &axisAnalysisPass,
                              PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferAtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value cmp = op.getCmp();
    Value val = op.getVal();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llVal = adaptor.getVal();
    Value llCmp = adaptor.getCmp();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = val.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    // Max supported vectorization for i32 and i64 is 1x
    // on CDNA3 and CDNA4
    // BUFFER_ATOMIC_CMPSWAP(i32) and BUFFER_ATOMIC_CMPSWAP_X2(i64)
    unsigned vec = 1u;

    // Get the offsets, val, and cmp
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valElems = unpackLLElements(loc, llVal, rewriter);
    SmallVector<Value> cmpElems = unpackLLElements(loc, llCmp, rewriter);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    SmallVector<Value> loadedVals;

    // We need to manually emit memory fences (LLVM doesn't do this for buffer
    // ops)
    auto memOrdering = op.getSem();
    auto memScope = op.getScope();
    // Release Fence pre-atomic
    if (failed(emitFence(op, rewriter, loc, memOrdering, memScope,
                         true /*preAtomic*/))) {
      return failure();
    }

    mlir::Operation *lastCASOp;
    MLIRContext *ctx = rewriter.getContext();
    GCNBuilder waitcntBuilder;

    // Check if the op has users, if it does we set GLC=1, otherwise GLC=0
    auto opUsers = op.getResult().getUsers();
    auto hasUsers = !op.getResult().getUsers().empty();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);

    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      Value pred = threadPred;
      // Create the store val
      Value casStoreVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valElems, vecStart);
      // Create the cmp val
      Value casCmpVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          cmpElems, vecStart);

      Value loadVal =
          bufferEmitter.emitAtomicCAS(vecTy, rsrcDesc, offsetElems[vecStart],
                                      casCmpVal, casStoreVal, pred, hasUsers);
      // Track the last op, so we can emit a fenceop after the loop
      lastCASOp = loadVal.getDefiningOp();

      for (size_t ii = 0; ii < vec; ++ii) {
        Value vecIdx = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        Value loaded = b.extract_element(valueElemTy, loadVal, vecIdx);
        loadedVals.push_back(loaded);
      }
    } // end vec

    // Emit post-atomic acquire fence
    if (failed(emitFence(op, rewriter, lastCASOp->getLoc(), memOrdering,
                         memScope, false /*preAtomic*/))) {
      return failure();
    }

    finalizeTensorAtomicResults(op, dyn_cast<RankedTensorType>(valueTy),
                                rewriter, loadedVals, valueElemTy, b,
                                threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct BufferStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::amdgpu::BufferStoreOp>,
      public LoadStoreConversionBase {
  BufferStoreOpConversion(LLVMTypeConverter &converter,
                          const AMD::TargetInfo &targetInfo,
                          ModuleAxisInfoAnalysis &axisAnalysisPass,
                          PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::BufferStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    LLVM::AMD::BufferEmitter bufferEmitter(rewriter, loc, targetInfo);

    // original values
    Value ptr = op.getPtr();
    Value offset = op.getOffsets();
    Value mask = op.getMask();
    Value data = op.getValue();
    auto cacheMod = op.getCache();

    Value llPtr = adaptor.getPtr();
    Value llOffset = adaptor.getOffsets();
    Value llMask = adaptor.getMask();
    Value llData = adaptor.getValue();
    Value llStride = adaptor.getStride();

    // Determine the vectorization size
    Type valueTy = data.getType();
    Type valueElemTy =
        typeConverter->convertType(getElementTypeOrSelf(valueTy));
    Type ptrType = getPointerTypeWithShape(ptr, offset);

    unsigned numElems = getTotalElemsPerThread(ptrType);
    unsigned vec = getVectorSize(ptr, offset, axisAnalysisPass);

    // Get the offsets and value
    SmallVector<Value> offsetElems = unpackLLElements(loc, llOffset, rewriter);
    SmallVector<Value> valueElems = unpackLLElements(loc, llData, rewriter);

    // Get the mask
    SmallVector<Value> maskElems =
        getMaskElemsAndUpdateVeclen(rewriter, loc, llMask, mask, vec);

    Value rsrcDesc = bufferEmitter.createResourceDescriptor(llPtr, llStride);
    MLIRContext *ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto freeVarMasks = getFreeVariableMasks(valueTy);
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    uint32_t regMask = freeVarMasks[str_attr("reg")];
    for (size_t vecStart = 0; vecStart < numElems; vecStart += vec) {
      if (!isCanonicalIndex(vecStart, regMask)) {
        // Don't emit store ops for redundant elements within a thread
        continue;
      }

      Value pred =
          llMask ? b.and_(threadPred, maskElems[vecStart]) : threadPred;

      Type vecTy = LLVM::getVectorType(valueElemTy, vec);
      // Create the store val
      Value storeVal = packElementRangeIntoVector(
          rewriter, this->getTypeConverter(), loc, cast<VectorType>(vecTy),
          valueElems, vecStart);
      bufferEmitter.emitStore(rsrcDesc, offsetElems[vecStart], storeVal, pred,
                              cacheMod);
    } // end vec

    rewriter.eraseOp(op);
    return success();
  }
};

struct AtomicCASOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicCASOp>,
      public LoadStoreConversionBase {
  AtomicCASOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // extract relevant info from Module
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    MLIRContext *ctx = rewriter.getContext();
    Value ptr = op.getPtr();

    Value llPtr = adaptor.getPtr();
    Value llCmp = adaptor.getCmp();
    Value llVal = adaptor.getVal();

    // prep data by unpacking to get data ready
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    auto cmpElements = unpackLLElements(loc, llCmp, rewriter);
    auto valElements = unpackLLElements(loc, llVal, rewriter);

    auto memOrdering = op.getSem();
    auto atomicMemOrdering = getMemoryOrdering(memOrdering);
    if (!atomicMemOrdering)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory ordering");
    auto scope = op.getScope();
    auto scopeStr = getAMDGPUMemScopeStr(scope);
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unknown AMDGPU memory scope");

    // deal with tensor or scalar
    auto valueTy = op.getResult().getType();
    auto tensorTy = dyn_cast<RankedTensorType>(valueTy);
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : valueTy;
    auto valueElemNBits = valueElemTy.getIntOrFloatBitWidth();
    auto elemsPerThread = getTotalElemsPerThread(op.getVal().getType());
    // vec = 1 for scalar
    auto vec = getVectorSize(op.getPtr(), axisAnalysisPass);
    // tensor
    if (tensorTy) {
      auto valTy = cast<RankedTensorType>(op.getVal().getType());
      vec = std::min<unsigned>(vec, valTy.getElementType().isF16() ? 2 : 1);
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    SmallVector<Value> resultVals(elemsPerThread);

    // atomic ops
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      Value casVal = b.undef(vecTy);
      for (int ii = 0; ii < vec; ++ii) {
        Value iiVal = createIndexAttrConstant(
            rewriter, loc, getTypeConverter()->getIndexType(), ii);
        casVal = b.insert_element(vecTy, casVal, valElements[i + ii], iiVal);
      }

      Value casPtr = ptrElements[i];
      Value casCmp = cmpElements[i];
      casVal = valElements[i];

      // use op
      if (tensorTy) { // for tensor
        auto retType = vec == 1 ? valueElemTy : vecTy;
        // TODO: USE ATOMIC CAS OP on Tensor
        auto successOrdering = *atomicMemOrdering;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef(scopeStr.value()));

        // Extract the new_loaded value from the pair.
        Value ret = b.extract_val(valueElemTy, cmpxchg, i);

        for (int ii = 0; ii < vec; ++ii) {
          resultVals[i + ii] =
              vec == 1 ? ret
                       : b.extract_element(valueElemTy, ret, b.i32_val(ii));
        }
      } else { // for scalar
        // Build blocks to bypass the atomic instruction for ~rmwMask.
        auto *curBlock = rewriter.getInsertionBlock();
        auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
        auto *atomicBlock = rewriter.createBlock(
            curBlock->getParent(), std::next(Region::iterator(curBlock)));

        // Fill entry block with global memory barrier and conditional branch.
        rewriter.setInsertionPointToEnd(curBlock);
        auto tid = getThreadId(rewriter, loc);
        Value pred = b.icmp_eq(tid, b.i32_val(i));
        rewriter.create<LLVM::CondBrOp>(loc, pred, atomicBlock, endBlock);

        // Build main block with atomic_cmpxchg.
        rewriter.setInsertionPointToEnd(atomicBlock);

        auto successOrdering = LLVM::AtomicOrdering::acq_rel;
        auto failureOrdering = LLVM::AtomicOrdering::monotonic;
        auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
            loc, casPtr, casCmp, casVal, successOrdering, failureOrdering,
            StringRef("agent"));

        if (!op.getResult().use_empty()) {
          // Extract the new_loaded value from the pair.
          Value newLoaded = b.extract_val(valueElemTy, cmpxchg, 0);
          Value atomPtr =
              getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
          b.store(newLoaded, atomPtr);
        }

        rewriter.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

        // Build the last block: synced load from shared memory, exit.
        rewriter.setInsertionPointToStart(endBlock);

        if (op.getResult().use_empty()) {
          rewriter.eraseOp(op);
          return success();
        }

        GCNBuilder BuilderMemfenceLDS;
        BuilderMemfenceLDS.create<>("s_waitcnt lgkmcnt(0)")->operator()();
        BuilderMemfenceLDS.launch(rewriter, loc, void_ty(ctx));
        b.barrier();
        Value atomPtr =
            getSharedMemoryBase(loc, rewriter, targetInfo, op.getOperation());
        Value ret = b.load(valueElemTy, atomPtr);
        rewriter.replaceOp(op, {ret});
        return success();
      }
    }

    // FIXME: threadPred = b.true_val() is buggy
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, b.true_val(), targetInfo,
                                getTypeConverter());
    return success();
  }
};

bool supportsGlobalAtomicF16PackedAndDpp(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  default:
    break;
  }
  return false;
}

struct AtomicRMWOpConversion
    : public ConvertOpToLLVMPattern<triton::AtomicRMWOp>,
      public LoadStoreConversionBase {
  AtomicRMWOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        ModuleAxisInfoAnalysis &axisAnalysisPass,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit),
        LoadStoreConversionBase(targetInfo, axisAnalysisPass) {}

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    auto binOp = matchAtomicOp(op.getAtomicRmwOp());
    if (!binOp)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW operation");

    auto memOrder = getMemoryOrdering(op.getSem());
    if (!memOrder)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW memory order");

    auto scopeStr = getAMDGPUMemScopeStr(op.getScope());
    if (!scopeStr)
      return rewriter.notifyMatchFailure(op, "Unsupported RMW scope");

    auto emitter =
        LLVM::AMD::AtomicRMWEmitter(targetInfo, *binOp, *memOrder, *scopeStr);

    Value val = op.getVal();
    Value ptr = op.getPtr();
    Value opResult = op.getResult();
    auto atomicRmwAttr = op.getAtomicRmwOp();

    Value llPtr = adaptor.getPtr();
    Value llVal = adaptor.getVal();
    Value llMask = adaptor.getMask();

    auto valElements = unpackLLElements(loc, llVal, rewriter);
    auto ptrElements = unpackLLElements(loc, llPtr, rewriter);
    SmallVector<Value> maskElements;
    if (llMask)
      maskElements = unpackLLElements(loc, llMask, rewriter);

    auto tensorTy = dyn_cast<RankedTensorType>(opResult.getType());
    Type valueElemTy =
        tensorTy ? getTypeConverter()->convertType(tensorTy.getElementType())
                 : opResult.getType();

    int numElems = 1;
    // In the case of unpaired f16 elements utilize dpp instructions to
    // accelerate atomics. Here is an algorithm of lowering
    // tt::atomicRmwOp(%ptr, %val, %mask):
    // 0. Group thread by pairs. Master thread is (tid % 2 == 0);
    // 1. All the threads send %val to (tid - 1) thread via dppUpdateOp shl, so
    //    all the masters receive value from secondary threads;
    // 2. Take into account parity in the %mask value, build control flow
    //    structures according to it;
    // 3. Generate llvm::atomicRmwOp in the threads enabled by %mask value;
    // 4. All the threads send result of generated operation to (tid + 1) thread
    //    via dppUpdateOp shl, so all secondary thread also receive their
    //    result.
    //
    // This approach enables us to use half the active threads committing atomic
    // requests to avoid generating of code providing unified access to f16
    // element and reduce contention.
    bool applyPackingF16 = false;
    auto vec = getVectorSize(ptr, axisAnalysisPass);

    // CDNA3/CDNA4 arch allows to accelerate its atomics with LDS reduction
    // algorithm, which is only applicable for atomics with no return. Otherwise
    // we have to deal with an additional overhead.
    bool enableIntraWaveReduce =
        llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                           targetInfo.getISAFamily()) &&
        tensorTy && opResult.use_empty();

    // TODO: support data types less than 32 bits
    enableIntraWaveReduce &= valueElemTy.getIntOrFloatBitWidth() >= 32;

    if (tensorTy) {
      bool isF16Ty = valueElemTy.isF16() || valueElemTy.isBF16();
      unsigned availableVecSize = isF16Ty ? 2 : 1;
      vec = std::min<unsigned>(vec, availableVecSize);
      // Force F16 packing in the case it's not coming in as packed, but the
      // ISA can support packed atomic instructions.
      applyPackingF16 =
          supportsGlobalAtomicF16PackedAndDpp(targetInfo.getISAFamily()) &&
          vec == 1 && isF16Ty && atomicRmwAttr == RMWOp::FADD &&
          !enableIntraWaveReduce;
      numElems = tensorTy.getNumElements();

      auto threadOrder = getThreadOrder(tensorTy);
      unsigned contigWithinLanes =
          axisAnalysisPass.getAxisInfo(ptr)->getContiguity(threadOrder.front());
      enableIntraWaveReduce &= contigWithinLanes == 1;
    }

    auto vecTy = vec_ty(valueElemTy, vec);
    auto elemsPerThread = getTotalElemsPerThread(val.getType());

    auto freeVarMasks = getFreeVariableMasks(op.getPtr().getType());
    Value threadPred =
        emitRedundantThreadPredicate(freeVarMasks, rewriter, loc, targetInfo);
    auto tid = getThreadId(rewriter, loc);

    std::optional<Value> atomicSharedMemBase =
        op->hasAttr("allocation.offset")
            ? std::optional<Value>(getSharedMemoryBase(
                  loc, rewriter, targetInfo, op.getOperation()))
            : std::nullopt;

    SmallVector<Value> resultVals(elemsPerThread);
    for (size_t i = 0; i < elemsPerThread; i += vec) {
      // TODO: in case llMask is zero we can create only one branch for all
      // elemsPerThread.
      Value rmwMask = llMask ? b.and_(threadPred, maskElements[i]) : threadPred;
      if (applyPackingF16) {
        resultVals[i] = emitter.emitPairedAtomicForEvenTID(
            rewriter, ptrElements[i], valElements[i], rmwMask);
      } else {
        Value valElement;
        if (vec == 1) {
          valElement = valElements[i];
        } else {
          Value vecVal = b.undef(vecTy);
          for (size_t ii = 0; ii < vec; ++ii)
            vecVal = b.insert_element(vecTy, vecVal, valElements[i + ii],
                                      b.i32_val(ii));
          valElement = vecVal;
        }

        // If we have a single tl.atomic_rmw that is lowered into multiple
        // llvm.atomic_rmw, and we set the ordering for each to aql_rel (the
        // default if no sem value is explicitly set in the DSL level
        // tl.atomic_add. The llvm backend will insert extra buffer invalidates
        // and L2 write backs causing a perforance degration. To avoid this we
        // set the ordering to release for the first, acquire for the last, and
        // relaxed for anything in between so that only a single set of
        // buffer_inv and buffer_wbl2 instructions are inserted by the backend
        // for any "cluster" of atomic ops.
        if ((vec > 1 || elemsPerThread > 1) &&
            op.getSem() == MemSemantic::ACQUIRE_RELEASE) {
          if (i == 0) {
            // First
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::release);
          } else if (i == elemsPerThread - vec) {
            // Last
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::acquire);
          } else {
            // Middle
            emitter.setAtomicOrdering(LLVM::AtomicOrdering::monotonic);
          }
        }

        Value retVal =
            emitter.emitAtomicRMW(rewriter, ptrElements[i], valElement, rmwMask,
                                  atomicSharedMemBase, enableIntraWaveReduce);

        if (tensorTy) {
          for (int ii = 0; ii < vec; ++ii) {
            resultVals[i + ii] =
                vec == 1
                    ? retVal
                    : b.extract_element(valueElemTy, retVal, b.i32_val(ii));
          }
        } else {
          if (!atomicSharedMemBase.has_value()) {
            rewriter.eraseOp(op);
            return success();
          }
          Value atomPtr = *atomicSharedMemBase;
          b.barrier();
          Value ret = b.load(valueElemTy, atomPtr);

          rewriter.replaceOp(op, {ret});
          return success();
        }
      }
    }
    finalizeTensorAtomicResults(op, tensorTy, rewriter, resultVals, valueElemTy,
                                b, threadPred, targetInfo, getTypeConverter());
    return success();
  }
};

struct AsyncWaitOpConversion : public ConvertOpToLLVMPattern<AsyncWaitOp> {
  AsyncWaitOpConversion(LLVMTypeConverter &converter,
                        const AMD::TargetInfo &targetInfo,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit), targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(AsyncWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    switch (targetInfo.getISAFamily()) {
    case ISAFamily::CDNA1:
    case ISAFamily::CDNA2:
    case ISAFamily::CDNA3:
    case ISAFamily::CDNA4:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "Only supported on CDNA target architecture");
    }

    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    // global.load.lds uses vmcnt to synchronize
    // The rocdl op stores all available counters in a single int32 value (v).
    // The vmcnt (6 bits) is split into a lower 3:0 and higher 5:4 parts.
    // The lower part is stored in bits 3:0 of v and the higher part in bits
    // 15:14. We have to set all other bits in v to 1 to signal we are not
    // interested in those.

    // Clamp vmcnt to 6bits; a lower vmcnt will produce a conservative wait
    unsigned vmCnt = std::min(63u, op.getNum());

    // Extract low and high bits and combine while setting all other bits to 1
    unsigned lowBits = vmCnt & 0xF;
    unsigned highBits = vmCnt >> 4 << 14;
    unsigned otherCnts = ~0xC00F; // C00F has bits 15:14 and 3:0 set
    unsigned waitValue = lowBits | highBits | otherCnts;

    rewriter.create<ROCDL::SWaitcntOp>(loc, waitValue);

    // Drop the result AsyncToken
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }

private:
  const AMD::TargetInfo &targetInfo;
};

struct AsyncCommitGroupOpConversion
    : public ConvertOpToLLVMPattern<AsyncCommitGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AsyncCommitGroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Drop the result AsyncToken
    auto loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.replaceOp(op, b.i32_val(0));
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {
void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit) {
  patterns.add<AtomicCASOpConversion, AtomicRMWOpConversion, LoadOpConversion,
               StoreOpConversion, BufferLoadOpConversion,
               BufferLoadToLocalOpConversion, BufferStoreOpConversion,
               BufferAtomicRMWOpConversion, AsyncCopyGlobalToLocalOpConversion,
               BufferAtomicCASOpConversion>(typeConverter, targetInfo,
                                            axisInfoAnalysis, benefit);
  patterns.add<AsyncWaitOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<AsyncCommitGroupOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
