add_triton_library(ProtonTransforms
  InsertProtonRecordAsyncOps.cpp

  DEPENDS
  ProtonTransformsIncGen
  LINK_LIBS PUBLIC
  ProtonIR
  TritonIR
  TritonGPUIR
)
