#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <math.h>

#include <CL/opencl.h>

#include <spead_api.h>

#include "shared.h"

#define SIZE          1024
#define CL_SUCCESS    0

#define KERNELS_FILE  "/kernels.cl"

/*#define SPEAD_DATA_ID       0x1001*/ 
#define SPEAD_DATA_ID       0x0

struct sapi_o {
  cl_context       ctx;
  cl_command_queue cq;
  cl_program       p;
  cl_kernel        k;
  cl_mem           clin;
  cl_mem           clout;
  unsigned char    *out;
  uint64_t         olen;
};

void spead_api_destroy(void *data)
{
  struct sapi_o *a;

  a = data;
  if (a){
    if(a->clin)
      clReleaseMemObject(a->clin);

    if(a->clout)
      clReleaseMemObject(a->clout);
  
    if(a->out)
      free(a->out);

    destroy(&(a->k), &(a->ctx), &(a->cq), &(a->p));
    
    free(a);
  }
#ifdef DEBUG
  fprintf(stderr, "%s: done\n", __func__);   
#endif
}

void *spead_api_setup()
{
  struct sapi_o *a;

  a = malloc(sizeof(struct sapi_o));
  if (a == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: logic could not malloc api obj\n");
#endif
    return NULL;
  }

  if (setup_ocl(KERNELDIR KERNELS_FILE, &(a->ctx), &(a->cq), &(a->p)) != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "e: setup_ocl error\n");
#endif
    spead_api_destroy(a);
    return NULL;
  }

  //a->k = get_kernel("coherent_dedisperse", &(a->p));
  a->k = get_kernel("ct", &(a->p));
  if (a->k == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: get_kernel error\n");
#endif
    spead_api_destroy(a);
    return NULL;
  } 

  a->clin   = NULL;
  a->clout  = NULL;
  a->out    = NULL;
  
  return a;
}


int spead_api_callback(struct spead_item_group *ig, void *data)
{
  struct spead_api_item *itm;
  struct sapi_o *a;
  uint64_t off;

  cl_int err;
  cl_event evt;

  size_t workGroupSize[1];

  a = data;

  if (ig == NULL || a == NULL){
#ifdef DEBUG
    fprintf(stderr, "[%d] e: callback parameter error\n", getpid());
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
  
#if 0
  print_data(itm->i_data, sizeof(unsigned char)*itm->i_len);
#endif

  if (a->out == NULL || itm->i_len != a->olen){
    a->olen = itm->i_len;
    a->out  = realloc(a->out, sizeof(unsigned char)* a->olen);
    if (a->out == NULL){
#ifdef DEBUG
      fprintf(stderr, "e: logic cannot malloc output buffer\n");
#endif
      return -1;
    }
#ifdef DEBUG
    fprintf(stderr, "%s: created ouput buffer %ld bytes\n", __func__, a->olen);
#endif
  }

  /*TODO:FIX length for changing data size*/

  if (a->clin == NULL){
    a->clin  = clCreateBuffer(a->ctx, CL_MEM_READ_ONLY, sizeof(unsigned char)*itm->i_len, NULL, &err);
    if (err != CL_SUCCESS){
#ifdef DEBUG
      fprintf(stderr, "clCreateBuffer return %s\n", oclErrorString(err));
#endif
      return -1;
    }
#ifdef DEBUG
    fprintf(stderr, "%s: created device input buffer %ld bytes\n", __func__, itm->i_len);
#endif
  }

  if (a->clout == NULL){
    a->clout = clCreateBuffer(a->ctx, CL_MEM_WRITE_ONLY, sizeof(unsigned char)*itm->i_len, NULL, &err);
    if (err != CL_SUCCESS){
#ifdef DEBUG
      fprintf(stderr, "clCreateBuffer return %s\n", oclErrorString(err));
#endif
      return -1;
    }
#ifdef DEBUG
    fprintf(stderr, "%s: created device ouput buffer %ld bytes\n", __func__, itm->i_len);
#endif
  }
  
  err = clEnqueueWriteBuffer(a->cq, a->clin, CL_TRUE, 0, sizeof(unsigned char)*itm->i_len, itm->i_data, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueWriteBuffer returns %s\n", oclErrorString(err));
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: enqueue write buffer\n", __func__);
#endif

  clReleaseEvent(evt);

  err = clSetKernelArg(a->k, 0, sizeof(cl_mem), (void *) &(a->clin));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clSetKernelArg(a->k, 1, sizeof(cl_mem), (void *) &(a->clout));
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clSetKernelArg return %s\n", oclErrorString(err));
#endif
    return -1;
  }

  workGroupSize[0] = itm->i_len;
  
  err = clEnqueueNDRangeKernel(a->cq, a->k, 1, NULL, workGroupSize, NULL, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
    return -1;
  }

  clReleaseEvent(evt);

  clFinish(a->cq);

  err = clEnqueueReadBuffer(a->cq, a->clout, CL_TRUE, 0, sizeof(unsigned char)*a->olen, a->out, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clEnqueueReadBuffer returns %s\n", oclErrorString(err));
#endif
    return -1;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: enqueue read buffer\n", __func__);
#endif

  clReleaseEvent(evt);
  
#if 0
  print_data(a->out, sizeof(unsigned char)*itm->i_len);
#endif

  return 0;
}

#ifdef STANDALONE

int main(int argc, char *argv[])
{
  struct sapi_o *a;

  a = spead_api_setup();

  spead_api_destroy(a);
    
  
#if 0
  float2 a[SIZE];
  float2 c[SIZE];
  
  int i;

  size_t workGroupSize[1];
  
  for (i=0; i<SIZE; i++){
    a[i].x = 1.0f;
    a[i].y = 1.0f;
  }
  
  cl_in = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float2) * SIZE, a, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateBuffer returns %s\n", oclErrorString(err));
#endif

  cl_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float2) * SIZE, NULL, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateBuffer returns %s\n", oclErrorString(err));
#endif

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &cl_in);
#ifdef DEBUG
  fprintf(stderr, "clSetKernelArg returns %s\n", oclErrorString(err));
#endif

  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &cl_out);
#ifdef DEBUG
  fprintf(stderr, "clSetKernelArg returns %s\n", oclErrorString(err));
#endif

  workGroupSize[0] = SIZE;
  
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &event);

#ifdef DEBUG
  fprintf(stderr, "clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
  
  clReleaseEvent(event);

  clFinish(command_queue);

  err = clEnqueueReadBuffer(command_queue, cl_out, CL_TRUE, 0, sizeof(float2) * SIZE, &c, 0, NULL, &event);
#ifdef DEBUG
  fprintf(stderr, "clEnqueueReadBuffer returns %s\n", oclErrorString(err));
#endif

  clReleaseEvent(event);

  for (i=0; i<SIZE; i++){
#ifdef DEBUG
    //fprintf(stdout, "%f %c%fj\n", c[i].x, (c[i].y > 0)? '+':' ', c[i].y);
    fprintf(stdout, "%d %f\n", i, hypotf(c[i].x, c[i].y));
#endif
  }

#ifdef DEBUG
  fprintf(stdout, "e\n\n");
#endif

  for (i=0; i<SIZE; i++){
#ifdef DEBUG
    //fprintf(stdout, "%f %c%fj\n", c[i].x, (c[i].y > 0)? '+':' ', c[i].y);
    fprintf(stdout, "%d %f\n", i, atan2f(c[i].x, c[i].y));
#endif
  }

#ifdef DEBUG
  fprintf(stdout, "e\n");
#endif

#ifdef DEBUG
  fprintf(stderr, "\n");
#endif


#endif

  return 0;
}

#endif




#if 0
  for(i=0; i<SIZE; i++)
  {
    a[i] = 1.0f;
    b[i] = 1.0f;
    c[i] = 0.0f;
  }
  
  /*create a on the gpu and copy at the same time*/
  cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * SIZE, a, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateBuffer returns %s\n", oclErrorString(err));
#endif
  /*create b on the gpu but dont copy yet*/
  cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY                       , sizeof(float) * SIZE, NULL, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateBuffer returns %s\n", oclErrorString(err));
#endif
  cl_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY                      , sizeof(float) * SIZE, NULL, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateBuffer returns %s\n", oclErrorString(err));
#endif

  /*now copy b*/
  err = clEnqueueWriteBuffer(command_queue, cl_b, CL_TRUE, 0, sizeof(float) * SIZE, b, 0, NULL, &event);
#ifdef DEBUG
  fprintf(stderr, "clEnqueueWriteBuffer returns %s\n", oclErrorString(err));
#endif

  clReleaseEvent(event);

  /*kernel args*/
  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &cl_a);
#ifdef DEBUG
  fprintf(stderr, "clSetKernelArg returns %s\n", oclErrorString(err));
#endif
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &cl_b);
#ifdef DEBUG
  fprintf(stderr, "clSetKernelArg returns %s\n", oclErrorString(err));
#endif
  err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &cl_c);
#ifdef DEBUG
  fprintf(stderr, "clSetKernelArg returns %s\n", oclErrorString(err));
#endif
  
  clFinish(command_queue);

  if(cl_a)
    clReleaseMemObject(cl_a);
  if(cl_b)
    clReleaseMemObject(cl_b);
  if(cl_c)
    clReleaseMemObject(cl_c);

  cl_mem cl_a;
  cl_mem cl_b;
  cl_mem cl_c;
  float a[SIZE];
  float b[SIZE];
#endif

