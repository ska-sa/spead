__kernel void vector_add(__global const float *A, __global const float *B, __global float *C) 
{
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}

#define D        (4.148808*1000.0)
#define DM       4000.0 
#define CF       15.0

__kernel void coherent_dedisperse(__global const float2 *in, __global float2 *out)
{
  int n = get_global_id(0);
  
  float freq = (float) n;

  const float p = (2.0 * M_PI_F * D * DM * (freq * freq)) / ((freq+CF)*(CF*CF));

  out[n].x = (float) cos(p);
  out[n].y = (float) (-1) * sin(p);
 
}
