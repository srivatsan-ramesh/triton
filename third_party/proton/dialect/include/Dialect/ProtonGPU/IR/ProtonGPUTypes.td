#ifndef PROTONGPU_TYPES
#define PROTONGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"

class PTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<ProtonGPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def PTG_SegmentType : PTG_TypeDef<"Segment", "seg", []> {
  let summary = "A segment in the internal buffer";

  let description = [{
    `proton_gpu.segment` is a type returned by `PTG_SegmentBaseOp`.
    It is used to represent the location of a segment in the profiling buffer
    that we store our profiling events into. It is lowered to `i32` type. Here,
    `-1` means the segment location is invalid.
  }];

  let parameters = (ins
    GranularityAttr:$granularity,
    DenseI32ArrayAttr:$selectIds
  );
}

#endif
