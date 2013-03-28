#define D        (4.148808*1000.0)
#define DM       67.99
#define CF       1.5*1000.0*1000.0*1000.0 

__kernel void coherent_dedisperse(__global const float2 *in, __global float2 *out)
{
  int n = get_global_id(0);
  
  float freq = (float) n;

  const float p = (2.0 * M_PI_F * D * DM * (freq * freq)) / ((freq+CF)*(CF*CF));

  const float2 chirp = { 
    cos(p), 
    (-1) * sin(p) 
  };

  out[n].x = (float) (in[n].x * chirp.x) - (in[n].y * chirp.y);
  out[n].y = (float) (in[n].x * chirp.y) + (in[n].y * chirp.x);
  
}

__kernel void ct(__global const unsigned char *in, __global unsigned char *out)
{
  int i = get_global_id(0);

  out[i] = in[i] + (unsigned char)1;

}

__kernel void power(__global const float2 *in, __global float *out)
{
  int i = get_global_id(0);
  out[i] = hypot(in[i].x, in[i].y);
}

__kernel void phase(__global const float2 *in, __global float *out)
{
  int i = get_global_id(0);
  out[i] = atan2(in[i].y, in[i].x);
}

#define BYTES_PER_SAMPLE 2

__kernel void power_uint8_to_float(__global const char *in, __global float *out)
{
  int i = get_global_id(0)*BYTES_PER_SAMPLE;

  float x = (float) in[i];
  float y = (float) in[i+1];

  out[i]    = (float) hypot(x, y);
  out[i+1]  = (float) atan2(y, x);
}

