#define D        (4.148808*1000.0)
#define DM       5.0 
#define CF       15.0

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
