#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include <spead_api.h>

#include "shared.h"

#define KERNELS_FILE  "/radix2_po2_kernel.cl"

#define SPEAD_DATA_ID    0xb001

#define FOLD_WINDOW      500
#define FOLD_DEPTH       50


struct sapi_object {
  struct ocl_ds     *o_ds;
  struct ocl_kernel *o_power_phase;
  struct ocl_kernel *o_chd;
  struct ocl_kernel *o_folder;
  struct ocl_kernel *o_memset;
  cl_mem            o_in;
  cl_mem            o_fold_map;
  int               o_fold_id;
  int               o_fold_did;
  void              *o_host;
  int               o_N;
};

int run_clmemset(struct sapi_object *so, struct ocl_kernel *k);

void destroy_sapi_object(void *data)
{
  struct sapi_object *so;
  so = data;
  if (so == NULL){
    destroy_ocl_mem(so->o_in);
    destroy_ocl_mem(so->o_fold_map);

    destroy_ocl_kernel(so->o_power_phase);
    destroy_ocl_kernel(so->o_chd);
    destroy_ocl_kernel(so->o_folder);
    destroy_ocl_kernel(so->o_memset);

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
  so->o_power_phase = NULL;
  so->o_N           = 0;
  so->o_in          = NULL;
  so->o_fold_map    = NULL;
  so->o_fold_id     = 0;
  so->o_fold_did    = 0;
  
  so->o_ds = create_ocl_ds(KERNELDIR KERNELS_FILE);
  if (so->o_ds == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_power_phase = create_ocl_kernel(so->o_ds, "power_phase");
  if (so->o_power_phase == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_chd = create_ocl_kernel(so->o_ds, "coherent_dedisperse");
  if(so->o_chd == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_folder = create_ocl_kernel(so->o_ds, "folder");
  if (so->o_folder == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_memset = create_ocl_kernel(so->o_ds, "clmemset");
  if (so->o_folder == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  return so;
}

int setup_cl_mem_buffers(struct sapi_object *so, int64_t len)
{
  if (so == NULL || len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: params so (%p) len %ld\n", __func__, so, len);
#endif
    return -1;
  }

  so->o_N = len;

  so->o_host = malloc(sizeof(float2)*len*FOLD_WINDOW);
  if (so->o_host == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: host memory creation failed\n", __func__);
#endif
    return -1;
  }

  bzero(so->o_host, sizeof(float2)*len);

  so->o_in = create_ocl_mem(so->o_ds, sizeof(float2)*len);
  if (so->o_in == NULL){
    free(so->o_host);
#ifdef DEBUG
    fprintf(stderr, "%s: device mem in creation failed\n", __func__);
#endif
    return -1;
  }

  so->o_fold_map = create_ocl_mem(so->o_ds, sizeof(float2)*len*FOLD_WINDOW);
  if (so->o_in == NULL){
    free(so->o_host);
    destroy_ocl_mem(so->o_in);
#ifdef DEBUG
    fprintf(stderr, "%s: device mem in creation failed\n", __func__);
#endif
    return -1;
  }

  if (run_clmemset(so, so->o_memset) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run memset fail\n", __func__);
#endif
    return -1;
  }


  return 0;
}

int run_chd(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[2], localz[2];
  cl_int err;
  cl_event evt;
  struct ocl_ds *ds;

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

  err  = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_in));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clSetKernelArg return %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }
  
  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueNDRangeKernel: %s\n", __func__, oclErrorString(err));
#endif

    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err  = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: cl profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  float run_time = (float)(ev_end_time - ev_start_time)/1000;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%f usec\033[0m\n", __func__, run_time);
#endif

  clReleaseEvent(evt);

  return 0;
}

int run_power_phase(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[2], localz[2];
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

#if 0
  workGroupSize[0] = (int) ceil(sqrt(so->o_N));
  workGroupSize[1] = (int) ceil(sqrt(so->o_N));
  localz[0] = LOZ;
  localz[1] = LOZ;
#endif
  workGroupSize[0] = so->o_N;

  err  = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_in));
  err |= clSetKernelArg(k->k_kernel, 1, sizeof(int), (void *) &(so->o_N));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clSetKernelArg return %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  //err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 2, NULL, workGroupSize, localz, 0, NULL, &evt);
  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueNDRangeKernel: %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err  = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: cl profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  float run_time = (float)(ev_end_time - ev_start_time)/1000;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%f usec\033[0m\n", __func__, run_time);
#endif

  clReleaseEvent(evt);

  return 0;
}

int run_folder(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[2], localz[2];
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

#if 0
  workGroupSize[0] = (int) ceil(sqrt(so->o_N));
  workGroupSize[1] = (int) ceil(sqrt(so->o_N));
  localz[0] = LOZ;
  localz[1] = LOZ;
#endif
  workGroupSize[0] = so->o_N;

  err  = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_in));
  err |= clSetKernelArg(k->k_kernel, 1, sizeof(cl_mem), (void *) &(so->o_fold_map));
  err |= clSetKernelArg(k->k_kernel, 2, sizeof(int), (void *) &(so->o_fold_id));
  err |= clSetKernelArg(k->k_kernel, 3, sizeof(int), (void *) &(so->o_N));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clSetKernelArg return %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  //err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 2, NULL, workGroupSize, localz, 0, NULL, &evt);
  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueNDRangeKernel: %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err  = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: cl profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  float run_time = (float)(ev_end_time - ev_start_time)/1000;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%f usec\033[0m\n", __func__, run_time);
#endif

  clReleaseEvent(evt);


  return 0;
}

int run_clmemset(struct sapi_object *so, struct ocl_kernel *k)
{
  size_t workGroupSize[2], localz[2];
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

#if 0
  workGroupSize[0] = (int) ceil(sqrt(so->o_N));
  workGroupSize[1] = (int) ceil(sqrt(so->o_N));
  localz[0] = LOZ;
  localz[1] = LOZ;
#endif
  workGroupSize[0] = so->o_N*FOLD_WINDOW;

  err = clSetKernelArg(k->k_kernel, 0, sizeof(cl_mem), (void *) &(so->o_fold_map));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clSetKernelArg return %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  //err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 2, NULL, workGroupSize, localz, 0, NULL, &evt);
  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueNDRangeKernel: %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err  = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: cl profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  float run_time = (float)(ev_end_time - ev_start_time)/1000;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%f usec\033[0m\n", __func__, run_time);
#endif

  clReleaseEvent(evt);

  return 0;
}


int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct sapi_object *so;
  
  float2 *pow;

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
    if (setup_cl_mem_buffers(so, itm->io_size) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: buffer setup failed\n", __func__);
#endif
      return -1;
    }
  }

  /*copy in uint8*/
  if (xfer_to_ocl_mem(so->o_ds, itm->io_data, sizeof(float2) * itm->io_size, so->o_in) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: xfer to ocl error\n", __func__);
#endif
    return -1;
  }

#if 1
  /*run chd*/
  if (run_chd(so, so->o_chd) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run chd fail\n", __func__);
#endif
    return -1;
  }
#endif

 /*run the power phase*/
  if (run_power_phase(so, so->o_power_phase) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run power phase fail\n", __func__);
#endif
    return -1;
  }
 
  /*fold*/
  if (run_folder(so, so->o_folder) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: run power phase fail\n", __func__);
#endif
    return -1;
  }

#ifdef DEBUG 
  fprintf(stderr, "%s fold_id: %d fold_did: %d\n", __func__, so->o_fold_id, so->o_fold_did);
#endif

  so->o_fold_id++;

  if (so->o_fold_id == FOLD_WINDOW){
    
    so->o_fold_id = 0;
    so->o_fold_did++;

    if (so->o_fold_did == FOLD_DEPTH){
      
      so->o_fold_did = 0;
#if 1
      /*copy data out*/
      if (xfer_from_ocl_mem(so->o_ds, so->o_fold_map, sizeof(float2) * so->o_N * FOLD_WINDOW, so->o_host) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: xfer from ocl error\n", __func__);
#endif
        return -1;
      }

      fprintf(stdout, "set term x11 size 1280,720\nset view map\nsplot '-' matrix with image\n");

      int i,j;

      for (j=0; j<FOLD_WINDOW; j++){
        pow = (float2*) (so->o_host + sizeof(float2) * j * so->o_N);
        for (i=so->o_N/2; i<so->o_N; i++){
          fprintf(stdout,"%0.11f ",pow[i].x);
        }
        fprintf(stdout, "\n");
      }

      fprintf(stdout, "e\ne\n");

#endif

      if (run_clmemset(so, so->o_memset) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: run memset fail\n", __func__);
#endif
        return -1;
      }


    }
    
  }
  
  if (set_spead_item_io_data(itm, NULL, 0) < 0){
#ifdef DEBUG
    fprintf(stderr, "err: resetting io ptr\n");
#endif
    return -1;
  }

#if 0
      /*copy data out*/
      if (xfer_from_ocl_mem(so->o_ds, so->o_in, sizeof(float2) * so->o_N, so->o_host) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: xfer from ocl error\n", __func__);
#endif
        return -1;
      }

      if (set_spead_item_io_data(itm, so->o_host, so->o_N) < 0){
        //if (set_spead_item_io_data(itm, so->o_in, so->o_N) < 0){
#ifdef DEBUG
        fprintf(stderr, "err: storeing cufft output\n");
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

