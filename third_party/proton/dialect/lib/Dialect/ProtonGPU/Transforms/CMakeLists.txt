add_triton_library(ProtonGPUTransforms
  ScheduleBufferStore.cpp

  DEPENDS
  ProtonGPUTransformsIncGen
  LINK_LIBS PUBLIC
  ProtonGPUIR
)
