



#include <cufft.h>




int main(int argc, char *argv[])
{
  
#define NX    256
#define BATCH 10

  cufftHandle    plan;
  cufftComplex  *data;
  cudaMalloc((void **) &data, sizeof(cufftComplex) * NX * BATCH);
  
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed to allocate\n");
#endif
    return 1;
  }

  if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr, "cuda err: plan creation failed\n");
#endif
    return 1;
  }

  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed");
#endif
    return 1;
  }

  if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed");
#endif
    return 1;
  }
  
  if (cudaThreadSynchronize() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed");
#endif
    return 1;
  }

  cufftDestroy(plan);
  cudaFree(data);

  return 0;
}
