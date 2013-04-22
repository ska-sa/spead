#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <spead_api.h>

#include "shared.h"

#define KERNELS_FILE  "/radix2_po2_kernel.cl"

#define SPEAD_DATA_ID    0xb001

struct sapi_object {
  struct ocl_ds     *o_ds;
  struct ocl_kernel *o_fft;
  struct ocl_kernel *o_fft_setup;
  struct ocl_kernel *o_u2f;
  cl_mem            o_map;
  cl_mem            o_in;
  cl_mem            o_out;
  float2            *o_host;

  int               o_N;
  int               o_threads;
  int               o_passes;
};

void destroy_sapi_object(void *data)
{
  struct sapi_object *so;
  so = data;
  if (so == NULL){
    destroy_ocl_mem(so->o_in);
    destroy_ocl_mem(so->o_out);
    destroy_ocl_mem(so->o_map);
    destroy_ocl_kernel(so->o_fft);
    destroy_ocl_kernel(so->o_u2f);
    destroy_ocl_kernel(so->o_fft_setup);
    destroy_ocl_ds(so->o_ds);
    if (so->o_host)
      free(so->o_host);
    free(so);
  }
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
  so->o_u2f         = NULL;
  so->o_fft_setup   = NULL;
  so->o_in          = NULL;
  so->o_out         = NULL;
  so->o_host        = NULL;
  so->o_map         = NULL;
  so->o_N           = 0;
  so->o_passes      = 0;
  so->o_threads     = 0;
  
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

  so->o_u2f = create_ocl_kernel(so->o_ds, "uint8_re_to_float2");
  if (so->o_fft == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_fft_setup = create_ocl_kernel(so->o_ds, "radix2_fft_setup");
  if (so->o_fft_setup == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: pid [%d] sapi obj (%p) FFT kernel (%p)\n", __func__, getpid(), so, so->o_fft);
#endif
  
  return so;
}




int setup_cl_mem_buffers(struct sapi_object *so, int64_t len)
{
  size_t workGroupSize[1];
  cl_int err;
  cl_event evt;
  struct ocl_kernel *k;
  struct ocl_ds     *ds;

  if (so == NULL || len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: params so (%p) len %ld\n", __func__, so, len);
#endif
    return -1;
  }

  k = so->o_fft_setup;
  ds= so->o_ds;

  if (is_power_of_2(len) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: %ld not a power of two\n", __func__, len);
#endif
    return -1;
  }

  so->o_N       = len;
  so->o_threads = len >> 1;
  so->o_passes  = power_of_2(len);
  if (so->o_passes == 0){
#ifdef DEBUG
    fprintf(stderr, "%s: need more data points\n", __func__);
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%ld bit FFT with %d threads and %d passes\033[0m\n", __func__, len, so->o_threads, so->o_passes);
#endif


  /*create the fft map*/
  so->o_map = create_ocl_mem(so->o_ds, sizeof(struct fft_map) * so->o_threads * so->o_passes);
  if (so->o_map == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: fft map memory creation failed\n", __func__);
#endif
    return -1;
  }



  workGroupSize[0] = so->o_threads;
  
  err = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_map));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clSetKernelArg(k->k_kernel, 1, sizeof(int), (void *) &(so->o_passes));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
    return -1;
  }

  clReleaseEvent(evt);
  clFinish(ds->d_cq);


#if 0
  so->o_host = malloc(sizeof(struct fft_map) * so->o_threads * so->o_passes);
#endif
  so->o_host = malloc(sizeof(float2)*len);
  if (so->o_host == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: host memory creation failed\n", __func__);
#endif
    destroy_ocl_mem(so->o_map);
    return -1;
  }

#if 0
  if (xfer_from_ocl_mem(ds, so->o_map, sizeof(struct fft_map) * so->o_threads * so->o_passes, so->o_host) == 0){
#ifdef DEBUG
    fprintf(stderr, "Printing out fft_map\n");
    struct fft_map fm;
    int t, p;
    for (t=0; t<so->o_threads; t++){
      for(p=0; p<so->o_passes; p++){
        fm = so->o_host[t*so->o_passes + p];
        fprintf(stderr, "A %d B %d W %d \t|", fm.A, fm.B, fm.W);
      }
      fprintf(stderr, "\n");
    }
#endif
  }
#endif

  so->o_in = create_ocl_mem(so->o_ds, sizeof(unsigned char)*len);
  if (so->o_in == NULL){
    free(so->o_host);
    destroy_ocl_mem(so->o_map);
#ifdef DEBUG
    fprintf(stderr, "%s: device mem in creation failed\n", __func__);
#endif
    return -1;
  }

  so->o_out = create_ocl_mem(so->o_ds, sizeof(float2)*len);
  if (so->o_out == NULL){
    free(so->o_host);
    destroy_ocl_mem(so->o_in);
    destroy_ocl_mem(so->o_map);
    so->o_in    = NULL;
    so->o_host  = NULL;
#ifdef DEBUG
    fprintf(stderr, "%s: device mem out creation failed\n", __func__);
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: cl_mem_buffers created\n", __func__);
#endif
  return 0;
}



int convert_real_to_float2(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[1];
  cl_int err;
  cl_event evt;
  struct ocl_ds     *ds;
  
  if (so == NULL || k == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: params\n", __func__);
#endif
    return -1;
  }
  
  ds = so->o_ds;
  if (ds == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: ds null\n", __func__);
#endif
    return -1;
  }
   
  workGroupSize[0] = so->o_N;
  
  err = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_in));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clSetKernelArg(k->k_kernel, 1, sizeof(cl_mem), (void *) &(so->o_out));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
    return -1;
  }

  clReleaseEvent(evt);
  clFinish(ds->d_cq);
  
  return 0;
}


int run_radix2_fft(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[1];
  cl_int err;
  cl_event evt;
  struct ocl_ds     *ds;
  
  if (so == NULL || k == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: params\n", __func__);
#endif
    return -1;
  }
  
  ds = so->o_ds;
  if (ds == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: ds null\n", __func__);
#endif
    return -1;
  }
   
  workGroupSize[0] = so->o_threads;
  
  err = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_map));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clSetKernelArg(k->k_kernel, 1, sizeof(cl_mem), (void *) &(so->o_out));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clSetKernelArg(k->k_kernel, 2, sizeof(const int), (void *) &(so->o_N));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }
  
  err = clSetKernelArg(k->k_kernel, 3, sizeof(const int), (void *) &(so->o_passes));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
    return -1;
  }

  clReleaseEvent(evt);
  clFinish(ds->d_cq);

  return 0;
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
  
  itm = get_spead_item_with_id(ig, SPEAD_DATA_ID);
  if (itm == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot find item with id 0x%x\n", __func__, SPEAD_DATA_ID);
#endif
    return -1;
  }

  if (so->o_map == NULL || so->o_in == NULL || so->o_out == NULL){
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



  /*copy in uint8*/
  if (xfer_to_ocl_mem(so->o_ds, itm->i_data, sizeof(unsigned char) * itm->i_data_len, so->o_in) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer to ocl error\n", __func__);
#endif
    return -1;
  }

  
  /*convert to float2*/
  if (convert_real_to_float2(so, so->o_u2f) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: convert to float2 fail\n", __func__);
#endif
    return -1;
  } 


  /*run radix2 fft*/
  if (run_radix2_fft(so, so->o_fft) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run fft fail\n", __func__);
#endif
    return -1;
  }

  

  /*copy data out*/
  if (xfer_from_ocl_mem(so->o_ds, so->o_out, sizeof(float2) * so->o_N, so->o_host) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer from ocl error\n", __func__);
#endif
    return -1;
  }

#ifdef DEBUG
  int i;
  for (i=0; i<so->o_N; i++){
    fprintf(stderr, "%f %f\n", so->o_host[i].x, so->o_host[i].y);
  }
#endif

#if 0
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
#endif


  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{

  return 0;
}


