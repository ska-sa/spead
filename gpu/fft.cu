/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>

#include <cufft.h>

#include <spead_api.h>

#define NX      1024
#define BATCH   1

#define SPEAD_DATA_ID       0x0


struct cufft_o {
  cufftHandle     plan;
  cufftComplex    *d_host;
  cufftComplex    *d_device;
};


void spead_api_destroy(void *data)
{
  struct cufft_o *fo;

  fo = data;

  if (fo){
  
    cufftDestroy(fo->plan);
    
    if (fo->d_host){
      free(fo->d_host);
    }
    
    cudaFree(fo->d_device);
  
  }
}


void *spead_api_setup()
{
  struct cufft_o *fo;
  
  fo = malloc(sizeof(struct cufft_o));
  if (fo == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: logic could not malloc api obj\n");
#endif
    return NULL;
  }
  
  fo->d_host  = NULL;
  fo->d_device = NULL;
  
  if (cufftPlan1d(&(fo->plan), NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr, "cuda err: plan creation failed\n");
#endif
    spead_api_destroy(fo);
    return NULL;
  }

  fo->d_host = (cufftComplex*) malloc(sizeof(cufftComplex) * NX * BATCH); 
  if (fo->d_host == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: malloc failed\n");
#endif
    spead_api_destroy(fo);
    return NULL;
  }
  
  cudaMalloc((void **) &(fo->d_device), sizeof(cufftComplex) * NX * BATCH);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed to cudamalloc\n");
#endif
    spead_api_destroy(fo);
    return NULL;
  }


  return fo;
}

int cufft_callback(struct cufft_o *fo, struct spead_api_item *itm)
{
  uint64_t i;

  uint8_t *d;  

  cufftComplex *hst;
  cufftComplex *dvc;

  cufftResult res;
  
  if (fo == NULL || itm == NULL){
    return -1;
  }
 
  if (NX*BATCH != itm->i_len){
#ifdef DEBUG
    fprintf(stderr, "e: data len [%ld] doesn't match fft setup NX*BATCH [%ld]\n", itm->i_len, (long int) NX*BATCH);
#endif
    return -1;
  }

  hst  = fo->d_host;
  dvc = fo->d_device;
  d   = itm->i_data;

  if (hst == NULL || dvc == NULL || d == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: data pointers are null\n");
#endif
    return -1;
  }

  /*prepare the data into cuComplex*/
  
  for (i=0; i<NX*BATCH; i++){
    hst[i].x = (float) d[i];
    hst[i].y = 0;
  }
  
  cudaMemcpy(dvc, hst, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed copy\n");
#endif
    return -1;
  }

  res = cufftExecC2C(fo->plan, dvc, dvc, CUFFT_FORWARD);
  if (res != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr ,"CUFFT error: ExecC2C Forward failed %d\n", res);
#endif
    return -1;  
  }

  cudaMemcpy(hst, dvc, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "cuda err: failed copy\n");
#endif
    return -1;
  }

  if (set_spead_item_io_data(itm, hst) < 0){
#ifdef DEBUG
    fprintf(stderr, "err: storeing cufft output\n");
#endif
    return -1;
  }

#if 0
  print_data( (unsigned char *) in, sizeof(cufftComplex)*NX*BATCH);
  for (i=0; i<NX*BATCH; i++){
    fprintf(stderr, "%f + j %f\n", in[i].x, in[i].y);
  }
#endif

#if 0
  /*compute power*/
  for (i=0; i<NX*BATCH; i++){
#ifdef DEBUG
    fprintf(stdout, "%ld %0.5f\n", i, cuCabsf(in[i]));
#endif
  }
#ifdef DEBUG
  fprintf(stdout, "e\n");
#endif
  /*compute phase*/
  for (i=0; i<NX*BATCH; i++){
#ifdef DEBUG
    fprintf(stdout, "%ld %0.5f\n", i, atan2(in[i].y, in[i].x));
#endif
  }
#ifdef DEBUG
  fprintf(stdout, "e\n");
#endif
#endif

  return 0;
}


int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct cufft_o *fo;
  uint64_t off;

  fo = data;

  if (fo == NULL || ig == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: NULL params for <%s>\n", __func__);
#endif
    return -1;
  }
  
  off = 0;
  while (off < ig->g_size){
    itm = (struct spead_api_item *) (ig->g_map + off);

#ifdef DEBUG
    fprintf(stderr, "ITEM id[0x%x] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
#endif
    if (itm->i_len == 0)
      goto skip;

    if (itm->i_id == SPEAD_DATA_ID){
      break;
    }
skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
  }

  if (itm->i_id != SPEAD_DATA_ID){
#ifdef DEBUG
    fprintf(stderr, "%s: err dont have requested data id\n", __func__);
#endif
    return -1;
  }
  

  if (cufft_callback(fo, itm) < 0){
    return -1;
  }


  return 0;
}


#if 0
#define PI (3.141592653589793)

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

#if 0
int main(int argc, char *argv[])
{
#define NX    256
#define BATCH 100

  int i; 

  cufftHandle    plan;
  cufftComplex  *odata;
  cufftComplex  *cxdata;
  cufftComplex  *redata;
  cufftComplex  *idata;

  redata = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*BATCH); 
  if (redata == NULL)
    return 1;

  cxdata = (cufftComplex*)malloc(sizeof(cufftComplex)*NX*BATCH); 
  if (cxdata == NULL)
    return 1;

  for (i=0; i<NX*BATCH; i++){
    redata[i].x = sinf(2 * PI * i / NX);
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

  cudaMalloc((void **) &idata, sizeof(cufftComplex) * NX * BATCH);
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

  cudaMemcpy(idata, redata, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyHostToDevice);
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

  if (cufftPlan1d(&plan, NX, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
#ifdef DEBUG
    fprintf(stderr, "cuda err: plan creation failed\n");
#endif
    cudaFree(idata);
    cudaFree(odata);
    free(redata);
    free(cxdata);
    return 1;
  }
  
  cufftResult res = cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);
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

  cudaMemcpy(cxdata, odata, sizeof(cufftComplex) * NX * BATCH, cudaMemcpyDeviceToHost);

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
    fprintf(stderr, "%d %0.5f %0.5f %0.5f\n", i, redata[i].x, cuCabsf(cxdata[i]), atan2(cxdata[i].y, cxdata[i].x));
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
#endif