#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <math.h>


#include <CL/opencl.h>

#include "shared.h"

#define SIZE          1024
#define CL_SUCCESS    0


int main(int argc, char *argv[])
{
  cl_context context;
  cl_command_queue command_queue;

  cl_program program;
  cl_kernel kernel;

  cl_int err;
  cl_event event;


  cl_mem cl_in;
  cl_mem cl_out;
  float2 c[SIZE];
  
  int i;

  size_t workGroupSize[1];


  if (setup_ocl("vadd.cl", &context, &command_queue, &program) != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "e: setup_ocl error\n");
#endif
    return 1;
  }

  kernel = get_kernel("coherent_dedisperse", &program);
  if (kernel == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: get_kernel error\n");
#endif
    return 1;
  } 
  
  cl_in = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float2) * SIZE, NULL, &err);
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

  if(cl_in)
    clReleaseMemObject(cl_in);
  if(cl_out)
    clReleaseMemObject(cl_out);
  
#if 0
  float a[SIZE];
  float b[SIZE];
#endif
  destroy(&kernel, &context, &command_queue, &program);


  return 0;
}


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
#endif

#if 0
  if(cl_a)
    clReleaseMemObject(cl_a);
  if(cl_b)
    clReleaseMemObject(cl_b);
  if(cl_c)
    clReleaseMemObject(cl_c);
#endif

#if 0
  cl_mem cl_a;
  cl_mem cl_b;
  cl_mem cl_c;
#endif
#if 0
  float a[SIZE];
  float b[SIZE];
#endif

