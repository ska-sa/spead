#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include <spead_api.h>

#define C_NUM   8192
#define B_ENG   1

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
  if (o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not malloc sapi object\n", __func__);
#endif
    return NULL;
  }

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

#ifdef DEBUG
    fprintf(stderr, "%s: sapi object destroyed\n", __func__);
#endif

  }
}

int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct sapi_obj *o;
  cudaError_t err;

  o = data;
  
  if (o == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error cannot use callback with null data\n", __func__);
#endif
    return -1; 
  }

#ifdef DEBUG
  fprintf(stderr, "%s: here\n", __func__);
#endif
  
#if 0
  if (o->o_size < ig->g_size){
#ifdef DEBUG
    fprintf(stderr, "%s: need more \n", __func__);
#endif
  }
#endif

  err = cudaMemcpy(o->o_data, ig->g_map, o->o_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess){
#ifdef DEBUG
    fprintf(stderr, "copy failed <%s>\n", __func__, cudaGetErrorString(err));
#endif
    return -1;  
  }
  
  
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

  return 0;
}


