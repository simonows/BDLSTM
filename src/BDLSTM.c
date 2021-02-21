#include "BDLSTM.cuh"

#ifdef __cplusplus
extern "C" {
#endif

int BDLSTM_GPU(
    double const **const input
  , double const* nnFlat
  , double const* nnLong
  , double const* nnShort
){
    return BDLSTM_cuda(input, nnFlat, nnLong, nnShort);
}


#ifdef __cplusplus
}
#endif

