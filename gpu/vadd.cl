__kernel void vector_add(__global const float *A, __global const float *B, __global float *C) 
{
  int i = get_global_id(0);
  C[i] = A[i] + B[i];
}

#define D        (4.148808*1000.0)
#define DM       50000000.0 
#define CF       150000.0

__kernel void coherent_dedisperse(__global const float2 *in, __global float2 *out)
{
  int n = get_global_id(0);
  
  float freq = (float) n;

  const float p = (2.0 * M_PI_F * D * DM * (freq * freq)) / ((freq+CF)*(CF*CF));

  const float2 chirp = { cos(p), (-1) * sin(p) };

  out[n].x = (float) (in[n].x * chirp.x) - (in[n].y * chirp.y);
  out[n].y = (float) (in[n].x * chirp.y) + (in[n].y * chirp.x);
 
}
