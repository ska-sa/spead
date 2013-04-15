#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#include "shared.h"

#define KERNELS_FILE  "/radix2_po2_kernel.cl"

#define SPEAD_BF_DATA_ID    0xb001

struct sapi_object {
  struct ocl_ds     *o_ds;
  struct ocl_kernel *o_fft;
  cl_mem            o_in;
  cl_mem            o_out;
  float             *o_host;
  uint64_t          o_len;
};

void destroy_sapi_object(void *data)
{
  struct sapi_object *so;
  so = data;
  if (so == NULL){
    destroy_ocl_mem(so->o_in);
    destroy_ocl_mem(so->o_out);
    destroy_ocl_kernel(so->o_fft);
    destroy_ocl_ds(so->o_ds);
    if (so->o_host)
      free(so->o_host);
    free(so);
  }
}

int setup_cl_mem_buffers(struct sapi_object *so, uint64_t len)
{
  if (so == NULL || len <= 0)
    return -1;

  so->o_host = malloc(sizeof(float)*len);
  if (so->o_host == NULL)
    return -1;
  
  so->o_in = create_ocl_mem(so->o_ds, sizeof(uint8_t)*len);
  if (so->o_in == NULL){
    free(so->o_host);
    return -1;
  }

  so->o_out = create_ocl_mem(so->o_ds, sizeof(float)*len);
  if (so->o_out == NULL){
    free(so->o_host);
    destroy_ocl_mem(so->o_in);
    so->o_in    = NULL;
    so->o_host  = NULL;
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: cl_mem_buffers created\n", __func__);
#endif
  return 0;
}

void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  destroy_sapi_object(data); 
}

void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct sapi_object *so;
  
  so = NULL;

  so = malloc(sizeof(struct sapi_object));
  if (so == NULL)
    return NULL;

  so->o_ds          = NULL;
  so->o_fft         = NULL;
  so->o_in          = NULL;
  so->o_out         = NULL;
  so->o_len         = 0;
  so->o_host        = NULL;
  
  so->o_ds = create_ocl_ds(KERNELDIR KERNELS_FILE);
  if (so->o_ds == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_fft = create_ocl_kernel(so->o_ds, "radix2_power_2_inplace_fft");
  if (so->o_fft == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: pid [%d] sapi obj (%p) kernel (%p)\n", __func__, getpid(), so, so->o_fft);
#endif
  
  return so;
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct sapi_object *so;
  
  itm = NULL;

  so = data;
  if (so == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: opencl not available\n", __func__);
#endif
    return -1;
  } 
  
  itm = get_spead_item_with_id(ig, SPEAD_BF_DATA_ID);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot find item with id 0x%x\n", __func__, SPEAD_BF_DATA_ID);
#endif
    return -1;
  }

  if (so->o_in == NULL || so->o_out == NULL){
    if (setup_cl_mem_buffers(so, itm->i_data_len) < 0)
      return -1;
  }
  
  if (xfer_to_ocl_mem(so->o_ds, itm->i_data, itm->i_data_len, so->o_in) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer to ocl error\n", __func__);
#endif
    return -1;
  }
#if 0
  if (run_1d_ocl_kernel(so->o_ds, so->o_power, itm->i_data_len / 2, so->o_in, so->o_out) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run ocl kernel error\n", __func__);
#endif
    return -1;
  }
#endif
  if (xfer_from_ocl_mem(so->o_ds, so->o_out, itm->i_data_len, so->o_host) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer to ocl error\n", __func__);
#endif
    return -1;
  }

  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{
  return 0;
}


