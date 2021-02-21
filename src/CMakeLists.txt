set(BDLSTM_SOURCES BDLSTM.cu BDLSTM.c)

add_library(BDLSTM_GPU SHARED ${BDLSTM_SOURCES})

# We need to explicitly state that we need all CUDA files in the
# particle library to be built with -dc as the member functions
# could be called by other libraries and executables
set_target_properties(BDLSTM_GPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

