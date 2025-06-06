#ifndef PROTONGPU_TRANSFORMS_PASSES
#define PROTONGPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ScheduleBufferStorePass: Pass<"proton-schedule-buffer-store", "mlir::ModuleOp"> {
  let summary = "Pass to move all Proton buffer stores to the end of the function";

  let description = "This pass makes the measurement more accurate by moving the expensive "
                    "shared memory stores to the end of the measured region after the measurements.";

  let dependentDialects = ["gpu::ProtonGPUDialect"];
}

def SyncWarpGroupDotPass: Pass<"sync-warp-group-dot", "mlir::ModuleOp"> {
  let summary = "Pass to sync warp group dot operations";
  let description = "This pass makes the warp group dot operations synchronouse "
                    "by making the wait operation to wait for the previous warp group "
                    "dot operation to finish.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif  // PROTONGPU_TRANSFORMS_PASSES
