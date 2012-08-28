#include <stdio.h>
#include <math.h>

#include <cufft.h>
#define PI (3.141592653589793)

#if 0
__global__ void real2complex(float *f, cufftComplex *fc, int N)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  int index = j*N+i;

  if (i<N && j<N) {
    fc[index].x = f[index];
    fc[index].y = 0.0f;
  }
}

__global__ void complex2real(cufftComplex *fc, float *f, int N)
{
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  int index = j*N+i;
  
  if (i<N && j<N) {
    f[index] = fc[index].x/((float)N*(float)N);
    //divide by number of elements to recover value
  }
}
#endif

int main(int argc, char *argv[])
{
#define NX    256
#define BATCH 10

  int i; 

  cufftHandle    plan;
  cufftComplex  *odata;
  cufftComplex  *cxdata;
  cufftReal     *redata;
  cufftReal     *idata;

  redata = (cufftReal*)malloc(sizeof(cufftReal)*NX*BATCH); 
  if (redata == NULL)
    return 1;

  cxdata = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*BATCH); 
  if (cxdata == NULL)
    return 1;

  for (i=0; i<NX*BATCH; i++){
    redata[i] = cosf(2 * PI * i / NX * BATCH);
  }

#if 0
  for (i=0; i<NX*BATCH; i++){
#ifdef DEBUG
    fprintf(stderr, "%0.2f ", redata[i]);
#endif
  }
#ifdef DEBUG
  fprintf(stderr, "\n");
#endif
#endif

  cudaMalloc((void **) &idata, sizeof(cufftReal) * NX * BATCH);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed to allocate\n");
#endif
    free(redata);
    free(cxdata);
    return 1;
  }

  cudaMalloc((void **) &odata, sizeof(cufftComplex) * NX * BATCH);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed to allocate\n");
#endif
    cudaFree(idata);
    free(redata);
    free(cxdata);
    return 1;
  }

  cudaMemcpy(idata, redata, sizeof(cufftReal) * NX * BATCH, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed copy\n");
#endif
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }

  if (cufftPlan1d(&plan, NX, CUFFT_R2C, BATCH) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr, "cuda err: plan creation failed\n");
#endif
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }
  
  cufftResult res = cufftExecR2C(plan, idata, odata);
  if (res != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed %d\n", res);
#endif
    cufftDestroy(plan);
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }

  cudaMemcpy(cxdata, odata, sizeof(cufftReal) * NX * BATCH, cudaMemcpyDeviceToHost);


#if 0
  if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed");
#endif
    cufftDestroy(plan);
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }
#endif
  
  if (cudaThreadSynchronize() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed");
#endif
    cufftDestroy(plan);
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }


  for (i=0; i<NX*BATCH; i++){
#ifdef DEBUG
    fprintf(stderr, "%d: %0.2f %0.2f\n", i, cxdata[i].x, cxdata[i].y);
#endif
  }
#ifdef DEBUG
  fprintf(stderr, "\n");
#endif


  cufftDestroy(plan);
  cudaFree(idata);
  cudaFree(odata);
  free(redata);
  free(cxdata);

  return 0;
}
