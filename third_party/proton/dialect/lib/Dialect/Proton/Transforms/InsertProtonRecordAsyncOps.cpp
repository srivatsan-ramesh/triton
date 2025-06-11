#include "Dialect/Proton/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include "Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::proton {

#define GEN_PASS_DEF_INSERTPROTONRECORDASYNCOPSPASS
#include "Dialect/Proton/Transforms/Passes.h.inc"

struct InsertProtonRecordAsyncOpsPass
    : public impl::InsertProtonRecordAsyncOpsPassBase<
          InsertProtonRecordAsyncOpsPass> {

  using impl::InsertProtonRecordAsyncOpsPassBase<
      InsertProtonRecordAsyncOpsPass>::InsertProtonRecordAsyncOpsPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    m.dump();
  }
};

} // namespace mlir::triton::proton
