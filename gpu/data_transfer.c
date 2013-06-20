#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#include "shared.h"

#define KERNELS_FILE  "/radix2_po2_kernel.cl"

#define SPEAD_DATA_ID    0xb001

struct sapi_obj {
  struct ocl_ds     *o_ds;
  cl_mem            o_in;
  void              *o_host;
  int               o_N;
};

void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  struct sapi_obj *o;
  o = data;
  if (o){
    destroy_ocl_mem(o->o_in);
    destroy_ocl_ds(o->o_ds);
    free(o);
  }
}

void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct sapi_obj *o;

  o = malloc(sizeof(struct sapi_obj));
  if (o == NULL)
    return NULL;
  
  o->o_ds = create_ocl_ds(KERNELDIR KERNELS_FILE);
  if (o->o_ds == NULL){
    spead_api_destroy(s, o);
    return NULL;
  }
  
  o->o_in   = NULL;
  o->o_host = NULL;
  o->o_N    = 0;

  return o;
}


int setup_cl_mem_buffers(struct sapi_obj *so, int64_t len)
{
  if (so == NULL || len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: params so (%p) len %ld\n", __func__, so, len);
#endif
    return -1;
  }

  so->o_N = len;

  so->o_host = malloc(len);
  if (so->o_host == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: host memory creation failed\n", __func__);
#endif
    return -1;
  }

  bzero(so->o_host, len);

  so->o_in = create_ocl_mem(so->o_ds, len);
  if (so->o_in == NULL){
    free(so->o_host);
#ifdef DEBUG
    fprintf(stderr, "%s: device mem in creation failed\n", __func__);
#endif
    return -1;
  }

  
  return 0;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct sapi_obj *so;

  itm = NULL;

  so = data;
  if (so == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: opencl not available\n", __func__);
#endif
    return -1;
  } 

  itm = get_spead_item_with_id(ig, SPEAD_DATA_ID);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot find item with id 0x%x\n", __func__, SPEAD_DATA_ID);
#endif
    return -1;
  }

  if (so->o_in == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: buffers about to be created\n", __func__);
#endif
    if (setup_cl_mem_buffers(so, itm->i_data_len) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: buffer setup failed\n", __func__);
#endif
      return -1;
    }
  }

  if (xfer_to_ocl_mem(so->o_ds, itm->i_data, so->o_N, so->o_in) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer to ocl error\n", __func__);
#endif
    return -1;
  }

  if (xfer_from_ocl_mem(so->o_ds, so->o_in, so->o_N, so->o_host) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer from ocl error\n", __func__);
#endif
    return -1;
  }
  

  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{

  return 0;
}
