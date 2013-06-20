#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <stdarg.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <stdarg.h>

#include <CL/opencl.h>

#include "shared.h"
#include <spead_api.h>

cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
  char            chBuffer[1024];
  cl_uint         num_platforms;
  cl_platform_id* clPlatformIDs;
  cl_int          ciErrNum;
  cl_uint         i = 0;

  *clSelectedPlatformID = NULL;

  // Get OpenCL platform count
  ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
  if (ciErrNum != CL_SUCCESS)
  {
    //shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
#ifdef DEBUG
    fprintf(stderr, " Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
#endif
    return -1000;
  }
  else
  {
    if(num_platforms == 0)
    {
      //shrLog("No OpenCL platform found!\n\n");
#ifdef DEBUG
      fprintf(stderr, "No OpenCL platform found!\n\n");
#endif
      return -2000;
    }
    else
    {
      // if there's a platform or more, make space for ID's
      if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
      {
        //shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
#ifdef DEBUG
        fprintf(stderr, "Failed to allocate memory for cl_platform ID's!\n\n");
#endif
        return -3000;
      }

      // get platform info for each platform and trap the NVIDIA platform if found
      ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
#if DEBUG>1
      fprintf(stderr, "Available platforms:\n");
#endif
      for(i = 0; i < num_platforms; ++i)
      {
        ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
        if(ciErrNum == CL_SUCCESS)
        {
#ifdef DEBUG
          fprintf(stderr, "platform %d: %s\n", i, chBuffer);
#endif
          if(strstr(chBuffer, "NVIDIA") != NULL)
          {
#if DEBUG>1
            fprintf(stderr, "++selected platform %d\n", i);
#endif
            *clSelectedPlatformID = clPlatformIDs[i];
            break;
          }
        }
      }

      // default to zeroeth platform if NVIDIA not found
      if(*clSelectedPlatformID == NULL)
      {
        //shrLog("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
        //printf("WARNING: NVIDIA OpenCL platform not found - defaulting to first platform!\n\n");
#ifdef DEBUG
        fprintf(stderr, "--selected platform: %d\n", 0);
#endif
        *clSelectedPlatformID = clPlatformIDs[0];
      }

      free(clPlatformIDs);
    }
  }

  return CL_SUCCESS;
}


// Helper function to get error string
// *********************************************************************
const char* oclErrorString(cl_int error)
{
  static const char* errorString[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
  };

  const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

  const int index = -error;

  return (index >= 0 && index < errorCount) ? errorString[index] : "";

}

int setup_ocl(char *kf, cl_context *context, cl_command_queue *command_queue, cl_program *program)
{
  cl_platform_id platform;
  cl_device_id *devices;
  cl_uint numDevices;
  cl_build_status build_status;

  cl_int err;

  size_t ret_val_size;

  char name[100];

  char *fc;
  int fd, i, j;
  struct stat fs;

  if (kf == NULL || context == NULL || command_queue == NULL || program == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: %s usage\n", __func__);
#endif
    return -1;
  }

  if (stat(kf, &fs) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: stat error: %s\n", strerror(errno));
#endif
    return -1;
  }
  
  fd = open(kf, O_RDONLY);
  if (fd < 0){
#ifdef DEBUG
    fprintf(stderr, "e: open error: %s\n", strerror(errno));
#endif
    return -1;
  }

  fc = mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (fc == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: mmap error: %s\n", strerror(errno));
#endif
    close(fd);
    return -1;
  }

  err = oclGetPlatformID(&platform);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "oclGetPlatformID returns %s\n", oclErrorString(err));
#endif
    return -1;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetDeviceIDs returns %s\n", oclErrorString(err));
#endif
    return -1;
  }
#ifdef DEBUG
  fprintf(stderr, "%s: have %d devices\n", __func__, numDevices);
#endif
 
  devices = malloc(sizeof(cl_device_id) * numDevices);
  if (devices == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: error malloc %s\n", strerror(errno));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    return -1;
  }

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetDeviceIDs returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

  for (i=0; i< numDevices; i++){
    err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), &name, NULL); 
    if (err != CL_SUCCESS){
      munmap(fc, fs.st_size);
      close(fd);
      free(devices);
      return -1;
    }
#ifdef DEBUG
    fprintf(stderr, "Device name: %s\n", name);
#endif

#if 1
#ifdef DEBUG

    cl_bool ecc;
    clGetDeviceInfo(devices[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, sizeof(ecc), &ecc, NULL);
    fprintf(stderr, "ECC: %d\n", ecc);

    cl_uint units;
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(units), &units, NULL); 
    fprintf(stderr, "Max clock frequency: %d\n", units);

    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(units), &units, NULL); 
    fprintf(stderr, "Max compute units (multiprocessors): %d\n", units);

    cl_ulong localmemsize;
    clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(localmemsize), &localmemsize, NULL); 
    fprintf(stderr, "Global memsize: %ld\n", localmemsize);

    clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localmemsize), &localmemsize, NULL); 
    fprintf(stderr, "Local memsize: %ld\n", localmemsize);

    size_t wgs;
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(wgs), &wgs, NULL); 
    fprintf(stderr, "Max work group size: %ld\n", wgs);

    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(units), &units, NULL); 
    fprintf(stderr, "Max work item dimentions: %d\n", units);

    size_t wid[units];
    clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t)*units, &wid, NULL); 
    fprintf(stderr, "Max work item sizes: ");

    for (j=0; j<units;j++)
      fprintf(stderr, "%ld ", wid[j]);
    fprintf(stderr, "\n");
#endif
#endif
  }



  *context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateContext returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

  int devid = 0;

  *command_queue = clCreateCommandQueue(*context, devices[devid], CL_QUEUE_PROFILING_ENABLE, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateCommandQueue returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  *program = clCreateProgramWithSource(*context, 1, (const char **) &fc, (const size_t *) &fs.st_size, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateProgramWithSource returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  err = clBuildProgram(*program, numDevices, devices, "-I /usr/local/cuda/include -I /opt/AMDAPP/include"
     "-cl-fast-relaxed-math -cl-single-precision-constant -cl-denorms-are-zero -cl-mad-enable -cl-no-signed-zeros ", NULL, NULL);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clBuildProgram returns %s\n", oclErrorString(err));
#endif

    err = clGetProgramBuildInfo(*program, devices[devid], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
#ifdef DEBUG
    fprintf(stderr, "clGetProgramBuildInfo BUILD STATUS %d\n", build_status);
#endif

    err = clGetProgramBuildInfo(*program, devices[devid], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);

    char build_log[ret_val_size+1];
    err = clGetProgramBuildInfo(*program, devices[devid], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

#ifdef DEBUG
    fprintf(stderr, "clGetProgramBuildInfo returns BUILD LOG\n\n%s\n", build_log);
#endif

    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

#if 0
def DEBUG
  fprintf(stderr, "%s: CL_DEVICE_MAX_WORK_GROUP_SIZE %ld\n", __func__, CL_DEVICE_MAX_WORK_GROUP_SIZE);
#endif

  munmap(fc, fs.st_size);
  close(fd);
  free(devices);

  return err;
}

cl_kernel get_kernel(char *name, cl_program *p)
{
  cl_kernel k;
  cl_int err;

  if (name == NULL || p == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: %s usage\n", __func__);
#endif
    return NULL;
  }

  k = clCreateKernel(*p, name, &err);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateKernel returns %s\n", oclErrorString(err));
#endif
    return NULL;
  }


#if 1 
#ifdef DEBUG
  size_t temp;
  clGetKernelWorkGroupInfo(k, NULL, CL_KERNEL_WORK_GROUP_SIZE, sizeof(temp), &temp, NULL);
  fprintf(stderr, "CL_KERNEL_WORK_GROUP_SIZE %ld\n", temp);
  clGetKernelWorkGroupInfo(k, NULL, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &temp, NULL);
  fprintf(stderr, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE %ld\n", temp);
#endif
#endif



  return k;
}

void destroy(cl_context *context, cl_command_queue *command_queue, cl_program *program)
{
  if(program)
    clReleaseProgram(*program);

  if(command_queue)
    clReleaseCommandQueue(*command_queue);

  if(context)
    clReleaseContext(*context);
}



#if 0
int compare_kernels(const void *v1, const void *v2)
{
  return strcmp((const char*) v1, (const char*) v2);
}
#endif

struct ocl_kernel* create_ocl_kernel(struct ocl_ds *d, char *kernel_name)
{
  struct ocl_kernel *k;

  if (kernel_name == NULL || d == NULL)
    return NULL;

  k = malloc(sizeof(struct ocl_kernel));
  if (k == NULL)
    return NULL;
  
  k->k_name = kernel_name;

  k->k_kernel = get_kernel(kernel_name, &(d->d_p));
  if (k->k_kernel == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: get_kernel error\n", __func__);
#endif
    free(k);
    return NULL;
  }

  return k;
} 


void destroy_ocl_kernel(void *data)
{
  struct ocl_kernel *k;
  k = data;
  if (k){
    
    if (k->k_kernel)
      clReleaseKernel(k->k_kernel);

    free(k);
  }
}

void destroy_ocl_ds(void *data)
{
  struct ocl_ds *ds;
  ds = data;
  if (ds){
    destroy(&(ds->d_ctx), &(ds->d_cq), &(ds->d_p));
#if 0
    destroy_avltree(ds->d_kernels, &destroy_ocl_kernel);
#endif
    free(ds);
  }
}

struct ocl_ds *create_ocl_ds(char *kernels_file)
{
  struct ocl_ds *ds;

  if (kernels_file == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot start opencl with null kernels path\n", __func__);
#endif
    return NULL;
  }

  ds = malloc(sizeof(struct ocl_ds));
  if (ds == NULL)
    return NULL;

  ds->d_ctx       = NULL;
  ds->d_cq        = NULL;
  ds->d_p         = NULL;
#if 0
  ds->d_kernels   = NULL;
#endif

  if (setup_ocl(kernels_file, &(ds->d_ctx), &(ds->d_cq), &(ds->d_p)) != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: setup_ocl error\n", __func__);
#endif
    destroy_ocl_ds(ds);
    return NULL;
  }

#if 0
  ds->d_kernels = create_avltree(&compare_kernels);
  if (ds->d_kernels == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: create kernel tree error\n");
#endif
    destroy_ocl_ds(ds);
    return NULL;
  }
#endif

  return ds;
}

cl_mem create_ocl_mem(struct ocl_ds *ds, size_t size)
{
  cl_mem m;
  cl_int err;

  if (ds == NULL || size <= 0){
#ifdef DEBUG 
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return NULL;
  }

  m = clCreateBuffer(ds->d_ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: error creating cl read/write buffer\n", __func__);
#endif
    return NULL;
  }

  return m;
}

int xfer_to_ocl_mem(struct ocl_ds *ds, void *src, size_t size, cl_mem dst)
{
  cl_int err;
  cl_event evt;

  if (ds == NULL || src == NULL || dst == NULL || size <= 0){
#ifdef DEBUG 
    fprintf(stderr, "%s: param error ds (%p) src (%p) dst (%p) size %ld\n", __func__, ds, src, dst, size);
#endif
    return -1;
  }

   /*copy data in*/
  err = clEnqueueWriteBuffer(ds->d_cq, dst, CL_TRUE, 0, size, src, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueWriteBuffer returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: cl profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

#if 0
  float run_time = (float)(ev_end_time - ev_start_time)/1000;
  float bit_rate = size / (float) run_time * 10e-3;
#endif
  float run_time = (float)(ev_end_time - ev_start_time);
  float bit_rate = size / (float) run_time;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%ld bytes in %f usec bitrate %f B/ns\033[0m\n", __func__, size, run_time, bit_rate);
#endif

  clReleaseEvent(evt);

  return 0;
}

int xfer_from_ocl_mem(struct ocl_ds *ds, cl_mem src, size_t size, void *dst)
{
  cl_int err;
  cl_event evt;

  if (ds == NULL || src == NULL || dst == NULL || size <= 0){
#ifdef DEBUG 
    fprintf(stderr, "%s: param error ds(%p) src(%p) dst(%p) size[%ld]\n", __func__, ds, src, dst, size);
#endif
    return -1;
  }

  /*copy data out*/
  err = clEnqueueReadBuffer(ds->d_cq, src, CL_TRUE, 0, size, dst, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: clEnqueueReadBuffer returns %s\n", __func__, oclErrorString(err));
#endif
    return -1;
  }

  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err = clWaitForEvents(1, &evt);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);

  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "%s: profiling returns %s\n", __func__, oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

#if 0
  float run_time = (float)(ev_end_time - ev_start_time)/1000;
  float bit_rate = size / (float) run_time * 10e-3;
#endif

  float run_time = (float)(ev_end_time - ev_start_time);
  float bit_rate = size / (float) run_time;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%ld bytes in %f usec bitrate %f B/ns\033[0m\n", __func__, size, run_time, bit_rate);
#endif

  clReleaseEvent(evt);
  
  return 0;
}

void destroy_ocl_mem(cl_mem m)
{ 
  if (m){
    clReleaseMemObject(m);
  }
}

int load_kernel_parameters(struct ocl_kernel *k, va_list ap_list)
{
  void *ptr;
  size_t size;
  int i;

  cl_int err;

  size = 0;
  ptr  = NULL;
  i    = 0;

  if (k == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: need a kernel to load parameters too!\n", __func__);
#endif
    return -1;
  }
  
  do {

    ptr = va_arg(ap_list, void *);
    if (!ptr)
      break;

    size = va_arg(ap_list, size_t);
    if (!size){
#ifdef DEBUG
      fprintf(stderr, "%s: need parameter size > 0\n", __func__);
#endif
      return -1;
    }

    err = clSetKernelArg(k->k_kernel, i, size, ptr);
    if (err != CL_SUCCESS){
#ifdef DEBUG
      fprintf(stderr, "%s: error clSetKernelArg [%d]: %s\n", __func__, i, oclErrorString(err));
#endif
      return -1;
    }

#ifdef DEBUG
    fprintf(stderr, "%s: loaded arg[%d] size[%ld] ptr<%p>\n", __func__, i, size, ptr);
#endif

    i++;
  } while(size && ptr);

  return 0;
}

#if 1
int run_1d_ocl_kernel(struct ocl_ds *ds, struct ocl_kernel *k, size_t work_group_size, ...)
{
  size_t globalz[1];
  size_t localz[1];
  cl_int err;
  cl_event evt;
  va_list ap;

  if (ds == NULL || k == NULL)
    return -1;

  localz[0]  = 16;
  globalz[0] = work_group_size;

  va_start(ap, work_group_size);
  if (load_kernel_parameters(k, ap) < 0){
    va_end(ap);
    return -1;
  }
  va_end(ap);



#if 1
  err = clEnqueueNDRangeKernel(ds->d_cq, k->k_kernel, 1, NULL, globalz, localz, 0, NULL, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "error clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
    return -1;
  }
  clFinish(ds->d_cq);

  cl_ulong ev_start_time = (cl_ulong) 0;     
  cl_ulong ev_end_time   = (cl_ulong) 0;   

  err  = clWaitForEvents(1, &evt);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clWaitForEvents %s\n", oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  err  = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_start_time, NULL);
  err |= clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time, NULL);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetEventProfiling %s\n", oclErrorString(err));
#endif
    clReleaseEvent(evt);
    return -1;
  }

  float run_time = (float)(ev_end_time - ev_start_time)/1000;

#ifdef DEBUG
  fprintf(stderr, "%s: \033[32m%f usec\033[0m\n", __func__, run_time);
#endif

  clReleaseEvent(evt);
#endif

  return 0;
}
#endif



int is_power_of_2(int x)
{
  return (x != 0) ? ((x & (x-1)) == 0 ? 0 : -1) : -1;
}



int power_of_2(int x)
{
  int p=0;
  while (x > 1) {
    x = x >> 1;
    p++;
  }
  return p;
}

