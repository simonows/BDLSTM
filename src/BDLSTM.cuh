
#ifdef __cplusplus
extern "C" {
#endif

int BDLSTM_cuda(
    double const **const input
  , double const* nnFlat
  , double const* nnLong
  , double const* nnShort
);

#ifdef __cplusplus
}
#endif

