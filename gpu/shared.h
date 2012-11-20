#ifndef SHARED_H
#define SHARED_H

#include <CL/opencl.h>

struct float2 {
  float x;
  float y;
};

typedef struct float2 float2;


cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);

const char* oclErrorString(cl_int error);

int setup_ocl(char *kf, cl_context *context, cl_command_queue *command_queue, cl_program *program);

void destroy(cl_context *context, cl_command_queue *command_queue, cl_program *program);

cl_kernel get_kernel(char *name, cl_program *p);

#endif
