#include <stdio.h>

#include "shared.h"

#define KERNELS_FILE  "/radix2_po2_kernel.cl"
#define KERNELDIR     "./"
#define len 100

int main(int argc, char *argv[])
{
  struct ocl_ds     *o_ds;
  cl_mem            o_in;
  cl_mem            o_out;
  
  
  o_ds = create_ocl_ds(KERNELDIR KERNELS_FILE);
  if (o_ds == NULL){
    return 1;
  }

  o_in = create_ocl_mem(o_ds, sizeof(unsigned char)*len);
  if (o_in == NULL){
    goto free_mem_in;
  }
  
  o_out = create_ocl_mem(o_ds, sizeof(float2)*len);
  if (o_out == NULL){
    goto free_mem_out;
  }







  destroy_ocl_mem(o_out);
free_mem_out:
  destroy_ocl_mem(o_in);
free_mem_in:
  destroy_ocl_ds(o_ds);

  return 0;
}
