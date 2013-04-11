



__kernel void radix2_dif_butterfly(__global const float2 A, __global const float2 B, __global const int k, __global const int N, __global float2 *X, __global float2 *Y)
{
  float2 x, y, z, w; 

  x.x = A.x + B.x;
  x.y = A.y + B.y;
  
  y.x = A.x - B.x;
  y.y = A.y - B.y;

  w.x = cos(2 * M_PI_F * k / N);
  w.y = (-1) * sin(2 * M_PI_F * k / N);

  z.x = y.x * w.x - y.y * w.y;
  z.y = y.y * w.x + y.x * w.y;
 
  *X.x = x.x;
  *X.y = x.y;
  *Y.x = z.x;
  *Y.y = z.y;
}

__kernel void radix2_power_2_inplace_fft(__global const float2 *in)
{
  

}
