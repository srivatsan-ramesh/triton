#include "Dialect/ProtonGPU/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton::gpu {

#define GEN_PASS_DEF_SCHEDULEBUFFERSTOREPASS
#define GEN_PASS_DEF_SYNCWARPGROUPDOTPASS
#include "Dialect/ProtonGPU/Transforms/Passes.h.inc"

struct ScheduleBufferStorePass
    : public impl::ScheduleBufferStorePassBase<ScheduleBufferStorePass> {

  using impl::ScheduleBufferStorePassBase<
      ScheduleBufferStorePass>::ScheduleBufferStorePassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = m.getContext();
    OpBuilder builder(context);

    // TODO(srir): Add support for non-inline kernels
    FuncOp func = *m.getOps<triton::FuncOp>().begin();
    auto startStoreList = llvm::SmallVector<CircularStoreOp, 8>();
    auto endStoreMap = llvm::SmallDenseMap<int, CircularStoreOp, 8>();

    func.walk([&](CircularStoreOp store) {
      if (store.getIsStart())
        startStoreList.push_back(store);
      else
        endStoreMap[store.getScopeId()] = store;
    });

    for (auto store : startStoreList) {
      int scopeId = store.getScopeId();
      auto endStore = endStoreMap[scopeId];
      if (!endStore) {
        mlir::emitError(func.getLoc(), "proton end store not found");
        signalPassFailure();
        return;
      }
      builder.setInsertionPoint(endStore);
      builder.clone(*store);
      store->erase();
    }
  }
};

class WarpGroupDotWaitReWrite
    : public mlir::OpRewritePattern<
          mlir::triton::nvidia_gpu::WarpGroupDotWaitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::triton::nvidia_gpu::WarpGroupDotWaitOp waitOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (waitOp.getPendings() > 0)
      rewriter.modifyOpInPlace(waitOp, [&]() {
        waitOp.setPendings(0);
        llvm::outs() << "waitOp pendings===================================: "
                     << waitOp.getPendings() << "\n";
      });
    return success();
  }
};

struct SyncWarpGroupDotPass
    : public impl::SyncWarpGroupDotPassBase<SyncWarpGroupDotPass> {
  using impl::SyncWarpGroupDotPassBase<
      SyncWarpGroupDotPass>::SyncWarpGroupDotPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    patterns.add<WarpGroupDotWaitReWrite>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir::triton::proton::gpu
