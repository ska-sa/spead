#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <sys/mman.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <spead_api.h>

#define C_NUM   8192
#define B_ENG   1


#define SPEAD_DATA_ID   0x0


struct sapi_obj {
  uint64_t o_size;
  void     *o_data;
};

void *spead_api_setup()
{
  struct sapi_obj *o;
  float2 *dd;
  cudaError_t err;

  o = malloc(sizeof(struct sapi_obj));
  //o = mmap(NULL, sizeof(struct sapi_obj), PROT_READ | PROT_WRITE | PROT_EXEC, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not malloc sapi object\n", __func__);
#endif
    return NULL;
  }

#if 0
  o->o_size = C_NUM*B_ENG * sizeof(float2);

  err = cudaMalloc((void **) &dd, o->o_size);
  if (err != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "%s: could not cudamalloc\n", __func__);
#endif
    free(o);
    return NULL;
  }

  o->o_data = dd;

#ifdef DEBUG
  fprintf(stderr, "%s: setup complete cudaMalloc %ld bytes\n", __func__, o->o_size);
#endif
#endif

  return o;
}

void spead_api_destroy(void *data)
{
  struct sapi_obj *o;

  o = data;

  if (o){
    
    if (o->o_data){
      cudaFree(o->o_data);
    }

    free(o);
    //munmap(o, sizeof(struct sapi_obj));

#ifdef DEBUG
    fprintf(stderr, "%s: sapi object destroyed\n", __func__);
#endif

  }
}

#if 0
int spead_worker_setup(void *data)
{
  struct sapi_obj *o;
  pid_t pid;
  float2 *dd;
  cudaError_t err;

  o = data;
  if (o == NULL)
    return -1;

  pid_t = getpid();

  err = cudaMalloc((void **) &dd, o->o_size);
  if (err != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "%s: could not cudamalloc\n", __func__);
#endif
    free(o);
    return NULL;
  }

  o->o_data = dd;

#ifdef DEBUG
  fprintf(stderr, "%s: setup complete cudaMalloc %ld bytes\n", __func__, o->o_size);
#endif



  return 0;
}
#endif

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct sapi_obj *o;
  cudaError_t err;
  //float2 *dd;
  uint8_t *dd;
  pid_t pid;

  o = data;
  pid = getpid();
  
  if (o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot use callback with null data\n", __func__);
#endif
    return -1; 
  }

#if 0
  dd = o->o_data;
  
  if (o->o_size < ig->g_size){
#ifdef DEBUG
    fprintf(stderr, "%s: need more \n", __func__);
#endif
  }
#endif

  err = cudaMalloc((void **) &dd, ig->g_size);
  if (err != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "copy failed %d <%s>\n", err, cudaGetErrorString(err));
#endif
    return -1;  
  }

#ifdef DEBUG
  fprintf(stderr, "~~~[%d]start copy %ld in cuda pointer @ (%p)\n", pid, ig->g_size, dd);
#endif

  err = cudaMemcpy(dd, ig->g_map, ig->g_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "copy failed %d <%s>\n", err, cudaGetErrorString(err));
#endif
    cudaFree(dd);
    return -1;  
  }
  
#ifdef DEBUG
  fprintf(stderr, "~~~[%d]end copy %ld in cuda pointer @ (%p)\n", pid, ig->g_size, dd);
#endif
  
#if 0
  uint64_t off = 0;
  uint64_t count;
  struct spead_api_item *itm;
  while (off < ig->g_size){
    itm = (struct spead_api_item *) (ig->g_map + off);
    if (itm->i_len == 0)
      goto skip;
    count = 0;
    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
    print_data(itm->i_data, itm->i_len);
skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
  }
#endif

  cudaFree(dd);

  return 0;
}


