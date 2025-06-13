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

// Generic templated function to insert RecordOp around any operator type
template <typename OpType>
void insertRecordAroundOp(OpType op, OpBuilder &builder, int scopeIndex) {
  // Get the operation name from its dialect and operation name
  std::string opTypeName = op->getName().getStringRef().str();

  // Create a unique name for this record operation
  std::string opName = opTypeName + "_" + std::to_string(scopeIndex);

  // Insert start record before the operation
  builder.setInsertionPoint(op);
  builder.create<mlir::triton::proton::RecordOp>(
      op.getLoc(),
      /*isStart=*/true,
      /*name=*/builder.getStringAttr(opName));

  // Insert end record after the operation
  builder.setInsertionPointAfter(op);
  builder.create<mlir::triton::proton::RecordOp>(
      op.getLoc(),
      /*isStart=*/false,
      /*name=*/builder.getStringAttr(opName));
}

// Generic templated function to walk through a module and insert RecordOp
// around specified operator types
template <typename OpType> void insertRecordAroundOpType(ModuleOp m) {
  MLIRContext *context = m.getContext();
  OpBuilder builder(context);
  int counter = 0;

  // Walk through all functions in the module
  m.walk([&](triton::FuncOp funcOp) {
    // Walk through all operations of the specified type in the function
    funcOp.walk(
        [&](OpType op) { insertRecordAroundOp(op, builder, counter++); });
  });
}

// Function to insert RecordOp at the start and end of a function
void insertRecordAroundFunction(triton::FuncOp funcOp, int &counter) {
  MLIRContext *context = funcOp->getContext();
  OpBuilder builder(context);

  // Get function name for the record operation
  std::string funcName = funcOp.getName().str();
  std::string recordName = funcName + "_" + std::to_string(counter++);

  // Insert start record at the beginning of the function
  if (!funcOp.empty()) {
    Block &entryBlock = funcOp.getBody().front();
    if (!entryBlock.empty()) {
      builder.setInsertionPointToStart(&entryBlock);
      builder.create<mlir::triton::proton::RecordOp>(
          funcOp.getLoc(),
          /*isStart=*/true,
          /*name=*/builder.getStringAttr(recordName));

      // Insert end record before the terminator of the last block
      Block &lastBlock = funcOp.getBody().back();
      if (!lastBlock.empty()) {
        Operation *terminator = lastBlock.getTerminator();
        if (terminator) {
          builder.setInsertionPoint(terminator);
          builder.create<mlir::triton::proton::RecordOp>(
              funcOp.getLoc(),
              /*isStart=*/false,
              /*name=*/builder.getStringAttr(recordName));
        }
      }
    }
  }
}

// Function to walk through a module and insert RecordOp at the start and end of
// all functions
void insertRecordAroundAllFunctions(ModuleOp m) {
  int counter = 0;

  // Walk through all functions in the module
  m.walk([&](triton::FuncOp funcOp) {
    insertRecordAroundFunction(funcOp, counter);
  });
}

struct InsertProtonRecordAsyncOpsPass
    : public impl::InsertProtonRecordAsyncOpsPassBase<
          InsertProtonRecordAsyncOpsPass> {

  using impl::InsertProtonRecordAsyncOpsPassBase<
      InsertProtonRecordAsyncOpsPass>::InsertProtonRecordAsyncOpsPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Insert RecordOp at the start and end of all functions
    insertRecordAroundAllFunctions(m);

    // Insert RecordOp around WarpGroupDotOp and WarpGroupDotWaitOp operations
    insertRecordAroundOpType<triton::nvidia_gpu::WarpGroupDotOp>(m);
    insertRecordAroundOpType<triton::nvidia_gpu::WarpGroupDotWaitOp>(m);

    // Insert RecordOp around AsyncCopyGlobalToLocalOp and AsyncWaitOp
    insertRecordAroundOpType<triton::gpu::AsyncCopyGlobalToLocalOp>(m);
    insertRecordAroundOpType<triton::gpu::AsyncWaitOp>(m);
  }
};

} // namespace mlir::triton::proton
