#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "proton/dialect/include/Dialect/Proton/Transforms/Passes.h"

#include "proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::proton {

#define GEN_PASS_DEF_INSERTPROTONRECORDASYNCOPSPASS
#include "proton/dialect/include/Dialect/Proton/Transforms/Passes.h.inc"

struct InsertProtonRecordAsyncOpsPass
    : public impl::InsertProtonRecordAsyncOpsPassBase<
          InsertProtonRecordAsyncOpsPass> {

  using impl::InsertProtonRecordAsyncOpsPassBase<
      InsertProtonRecordAsyncOpsPass>::InsertProtonRecordAsyncOpsPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();
    OpBuilder builder(context);
    int i = 0;

    // Walk through all functions in the module
    m.walk([&](triton::FuncOp funcOp) {
      // Walk through all WarpGroupDotWaitOp operations in the function
      funcOp.walk([&](triton::LoadOp loadOp) {
        // Create a unique name for this record operation
        std::string opName = "load_" + std::to_string(i++);

        // Insert start record before the wait operation
        builder.setInsertionPoint(loadOp);
        builder.create<mlir::triton::proton::RecordOp>(
            loadOp.getLoc(),
            /*isStart=*/true,
            /*name=*/builder.getStringAttr(opName));

        // Insert end record after the wait operation
        builder.setInsertionPointAfter(loadOp);
        builder.create<mlir::triton::proton::RecordOp>(
            loadOp.getLoc(),
            /*isStart=*/false,
            /*name=*/builder.getStringAttr(opName));
      });
    });
  }
};

} // namespace mlir::triton::proton
