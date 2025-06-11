#ifndef PROTON_TRANSFORMS_PASSES
#define PROTON_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def InsertProtonRecordAsyncOpsPass: Pass<"proton-insert-record-async-ops", "mlir::ModuleOp"> {
  let summary = "Pass to insert proton.record around async ops";
  
  let description = "This pass inserts proton.record around async ops to measure different async "
                    "ops issue times and wait times in the kernel.";

  let dependentDialects = ["mlir::triton::proton::ProtonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif  // PROTON_TRANSFORMS_PASSES
