#ifndef SHARED_H
#define SHARED_H

#include <CL/opencl.h>

struct float2 {
  float x;
  float y;
};
typedef struct float2 float2;


struct ocl_ds {
  cl_context       d_ctx;
  cl_command_queue d_cq;
  cl_program       d_p;
#if 0
  struct avl_tree  *d_kernels;
#endif
};

struct ocl_kernel {
  char *k_name;
  cl_kernel k_kernel;
};


cl_int oclGetPlatformID(cl_platform_id* clSelectedPlatformID);
const char* oclErrorString(cl_int error);

int setup_ocl(char *kf, cl_context *context, cl_command_queue *command_queue, cl_program *program);
void destroy(cl_context *context, cl_command_queue *command_queue, cl_program *program);
cl_kernel get_kernel(char *name, cl_program *p);


struct ocl_ds *create_ocl_ds(char *kernels_file);
void destroy_ocl_ds(void *data);
struct ocl_kernel* create_ocl_kernel(struct ocl_ds *d, char *kernel_name);
void destroy_ocl_kernel(void *data);

cl_mem create_ocl_mem(struct ocl_ds *ds, size_t size);
void destroy_ocl_mem(cl_mem m);

int xfer_to_ocl_mem(struct ocl_ds *ds, void *src, size_t size, cl_mem dst);
int xfer_from_ocl_mem(struct ocl_ds *ds, cl_mem src, size_t size, void *dst);
int run_1d_ocl_kernel(struct ocl_ds *ds, struct ocl_kernel *k, size_t work_group_size, ...);

#endif

