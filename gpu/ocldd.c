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



#define SIZE  100
#define CL_SUCCESS   0

//NVIDIA's code follows
//license issues probably prevent you from using this, but shouldn't take long
//to reimplement
//////////////////////////////////////////////////////////////////////////////
//! Gets the platform ID for NVIDIA if available, otherwise default to platform 0
//!
//! @return the id 
//! @param clSelectedPlatformID         OpenCL platform ID
//////////////////////////////////////////////////////////////////////////////
cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID)
{
  char chBuffer[1024];
  cl_uint num_platforms;
  cl_platform_id* clPlatformIDs;
  cl_int ciErrNum;
  *clSelectedPlatformID = NULL;
  cl_uint i = 0;

  // Get OpenCL platform count
  ciErrNum = clGetPlatformIDs (0, NULL, &num_platforms);
  if (ciErrNum != CL_SUCCESS)
  {
    //shrLog(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
    printf(" Error %i in clGetPlatformIDs Call !!!\n\n", ciErrNum);
    return -1000;
  }
  else
  {
    if(num_platforms == 0)
    {
      //shrLog("No OpenCL platform found!\n\n");
      printf("No OpenCL platform found!\n\n");
      return -2000;
    }
    else
    {
      // if there's a platform or more, make space for ID's
      if ((clPlatformIDs = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id))) == NULL)
      {
        //shrLog("Failed to allocate memory for cl_platform ID's!\n\n");
        printf("Failed to allocate memory for cl_platform ID's!\n\n");
        return -3000;
      }

      // get platform info for each platform and trap the NVIDIA platform if found
      ciErrNum = clGetPlatformIDs (num_platforms, clPlatformIDs, NULL);
      printf("Available platforms:\n");
      for(i = 0; i < num_platforms; ++i)
      {
        ciErrNum = clGetPlatformInfo (clPlatformIDs[i], CL_PLATFORM_NAME, 1024, &chBuffer, NULL);
        if(ciErrNum == CL_SUCCESS)
        {
          printf("platform %d: %s\n", i, chBuffer);
          if(strstr(chBuffer, "NVIDIA") != NULL)
          {
            printf("selected platform %d\n", i);
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
        printf("selected platform: %d\n", 0);
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



int main(int argc, char *argv[])
{
  cl_platform_id platform;
  cl_device_id *devices;
  cl_uint numDevices;
  cl_context context;

  cl_program program;
  cl_build_status build_status;
  size_t ret_val_size;

  cl_kernel kernel;
  cl_command_queue command_queue;

  cl_int err;
  cl_event event;

  cl_mem cl_a;
  cl_mem cl_b;
  cl_mem cl_c;
  
  float a[SIZE];
  float b[SIZE];
  float c[SIZE];
  
  int i;

  size_t workGroupSize[1];


  char *fc, *fpath = "vadd.cl";
  int fd;
  struct stat fs;

  if (stat(fpath, &fs) < 0){
#ifdef DEBUG
    fprintf(stderr, "stat error: %s\n", strerror(errno));
#endif
    return -1;
  }
  
  fd = open(fpath, O_RDONLY);
  if (fd < 0){
#ifdef DEBUG
    fprintf(stderr, "open error: %s\n", strerror(errno));
#endif
    return -1;
  }

  fc = mmap(NULL, fs.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (fc == NULL){
#ifdef DEBUG
    fprintf(stderr, "mmap error: %s\n", strerror(errno));
#endif
    return -1;
  }

  


  err = oclGetPlatformID(&platform);
#ifdef DEBUG
  fprintf(stderr, "oclGetPlatformID returns %s\n", oclErrorString(err));
#endif

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
#ifdef DEBUG
  fprintf(stderr, "clGetDeviceIDs returns %s\n", oclErrorString(err));
#endif
 
  devices = malloc(sizeof(cl_device_id) * numDevices);
  if (devices == NULL)
    return -1;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
#ifdef DEBUG
  fprintf(stderr, "clGetDeviceIDs returns %s\n", oclErrorString(err));
#endif



  context = clCreateContext(0, 1, devices, NULL, NULL, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateContext returns %s\n", oclErrorString(err));
#endif

  command_queue = clCreateCommandQueue(context, devices[0], 0, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateCommandQueue returns %s\n", oclErrorString(err));
#endif


  program = clCreateProgramWithSource(context, 1, (const char **) &fc, &fs.st_size, &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateProgramWithSource returns %s\n", oclErrorString(err));
#endif

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
#ifdef DEBUG
  fprintf(stderr, "clBuildProgram returns %s\n", oclErrorString(err));
#endif

  err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
#ifdef DEBUG
  fprintf(stderr, "clGetProgramBuildInfo returns %s BUILD STATUS %d\n", oclErrorString(err), build_status);
#endif


  err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
#ifdef DEBUG
  fprintf(stderr, "clGetProgramBuildInfo returns %s\n", oclErrorString(err));
#endif


  char build_log[ret_val_size+1];
  err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);
#ifdef DEBUG
  fprintf(stderr, "clGetProgramBuildInfo returns %s BUILD LOG [%s]\n", oclErrorString(err), build_log);
#endif


  kernel = clCreateKernel(program, "vector_add", &err);
#ifdef DEBUG
  fprintf(stderr, "clCreateKernel returns %s\n", oclErrorString(err));
#endif

  if (err != CL_SUCCESS){
    goto clean;
  }


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
  cl_c = clCreateBuffer(context, CL_MEM_READ_WRITE                      , sizeof(float) * SIZE, NULL, &err);
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



  workGroupSize[0] = 1;
  
  err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, workGroupSize, NULL, 0, NULL, &event);

#ifdef DEBUG
  printf("clEnqueueNDRangeKernel: %s\n", oclErrorString(err));
#endif
  
  clReleaseEvent(event);

  clFinish(command_queue);





  err = clEnqueueReadBuffer(command_queue, cl_c, CL_TRUE, 0, sizeof(float) * SIZE, &c, 0, NULL, &event);
#ifdef DEBUG
  fprintf(stderr, "clEnqueueReadBuffer returns %s\n", oclErrorString(err));
#endif

  clReleaseEvent(event);


  for (i=0; i<SIZE; i++){
#ifdef DEBUG
    fprintf(stderr, "%f ", c[i]);
#endif
  }
  
#ifdef DEBUG
  fprintf(stderr, "\n");
#endif


  if(cl_a)
    clReleaseMemObject(cl_a);
  if(cl_b)
    clReleaseMemObject(cl_b);
  if(cl_c)
    clReleaseMemObject(cl_c);
    
clean:
  if(program)
    clReleaseProgram(program);
  if(kernel)
    clReleaseKernel(kernel); 
  if(command_queue)
    clReleaseCommandQueue(command_queue);


  if(context)
    clReleaseContext(context);

  if(devices)
    free(devices);

  return 0;
}


