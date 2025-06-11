#ifndef PROTON_TRANSFORMS_PASSES_H_
#define PROTON_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::triton::proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/dialect/include/Dialect/Proton/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "proton/dialect/include/Dialect/Proton/Transforms/Passes.h.inc"

} // namespace mlir::triton::proton

#endif // PROTON_TRANSFORMS_PASSES_H_
