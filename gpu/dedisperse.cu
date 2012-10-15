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


#define SPEAD_DATA_ID       0x0    /*data id*/

#define DATA_CHANNELS       1024   /*2k fft with only positive results*/
#define DATA_CHANNEL_SETS   128    /*1 byte real 1 byte img*/


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

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct sapi_obj *o;
  uint64_t off;
  uint64_t count;
  cudaError_t err;
  uint8_t *dd;

  o = data;

  if (ig == NULL || o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot use callback with null data\n", __func__);
#endif
    return -1; 
  }

  off = 0;
  while (off < ig->g_size){
    itm = (struct spead_api_item *) (ig->g_map + off);

    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
    if (itm->i_len == 0)
      goto skip;

    count = 0;
#if 0
    fprintf(stderr, "ITEM id[%d] vaild [%d] len [%ld]\n", itm->i_id, itm->i_valid, itm->i_len);
    print_data(itm->i_data, itm->i_len);
#endif
    if (itm->i_id == SPEAD_DATA_ID)
      break;
skip:
    off += sizeof(struct spead_api_item) + itm->i_len;
  }


  if (itm && itm->i_id == SPEAD_DATA_ID){

#ifdef DEBUG
    fprintf(stderr, "SPEAD DATA ID FOUND\n");
#endif
      
    err = cudaMalloc((void **) &dd, itm->i_len);
    if (err != cudaSuccess){
#ifdef DEBUG
      fprintf(stderr, "copy failed %d <%s>\n", err, cudaGetErrorString(err));
#endif
      return -1;  
    }

#ifdef DEBUG
    fprintf(stderr, "CUDA malloc of size %ld\n", itm->i_len);
#endif

    err = cudaMemcpy(dd, itm->i_data, itm->i_len, cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
#ifdef DEBUG
      fprintf(stderr, "copy failed %d <%s>\n", err, cudaGetErrorString(err));
#endif
      cudaFree(dd);
      return -1;  
    }
  
    
    




    cudaFree(dd);

  }
  
  return 0;
}


