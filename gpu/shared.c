#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include <CL/opencl.h>

#include "shared.h"

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
#ifdef DEBUG
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
#ifdef DEBUG
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
  int fd;
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

  err = clGetDeviceInfo(devices[0], CL_DEVICE_NAME, sizeof(name), &name, NULL); 
  if (err != CL_SUCCESS){
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;

  }
#ifdef DEBUG
  fprintf(stderr, "Device name: %s\n", name);
#endif

  *context = clCreateContext(0, 1, devices, NULL, NULL, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateContext returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

  *command_queue = clCreateCommandQueue(*context, devices[0], 0, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateCommandQueue returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  *program = clCreateProgramWithSource(*context, 1, (const char **) &fc, &fs.st_size, &err);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clCreateProgramWithSource returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  err = clBuildProgram(*program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clBuildProgram returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

  err = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetProgramBuildInfo returns %s BUILD STATUS %d\n", oclErrorString(err), build_status);
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  err = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetProgramBuildInfo returns %s\n", oclErrorString(err));
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }


  char build_log[ret_val_size+1];
  err = clGetProgramBuildInfo(*program, devices[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
  if (err != CL_SUCCESS){
#ifdef DEBUG
    fprintf(stderr, "clGetProgramBuildInfo returns %s BUILD LOG [%s]\n", oclErrorString(err), build_log);
#endif
    munmap(fc, fs.st_size);
    close(fd);
    free(devices);
    return -1;
  }

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

  return k;
}

void destroy(cl_kernel *kernel, cl_context *context, cl_command_queue *command_queue, cl_program *program)
{
  if(program)
    clReleaseProgram(*program);

  if(kernel)
    clReleaseKernel(*kernel); 

  if(command_queue)
    clReleaseCommandQueue(*command_queue);

  if(context)
    clReleaseContext(*context);
}

