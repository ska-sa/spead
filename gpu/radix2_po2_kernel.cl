/**Author adam@ska.ac.za**/

void radix2_dif_butterfly(__global const float2 A, __global const float2 B, __global const int k, __global const int N, __global float2 *X, __global float2 *Y)
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

__kernel void radix2_power_2_inplace_fft(__global const float2 *in, __global const int N, __global const int passes)
{
  int a, b, w, p, t, m, groups, threads;

  threads = get_global_size(0);
  t = get_global_id(0);
  m = N >> 1;
  groups = threads / m;
  
  for (p=0; p<passes; p++){
    
    a = t + (t/m) * m;
    b = a + m;
    w = (t % m) * groups;
    
    radix2_dif_butterfly(in[a], in[b], w, N, &in[a], &in[b]);
      
    m = m >> 1;  
    groups = (m > 0) ? threads / m : 0;

    barrier(CLK_GLOBAL_MEM_FENCE);
  }

}
