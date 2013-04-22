/**Author adam@ska.ac.za**/

//#pragma OPENCL EXTENSION cl_khr_fp64: enable


struct fft_map {
  int A;
  int B;
  int W;
};


__kernel void radix2_fft_setup(__global struct fft_map *map, const int passes)
{
  register int t, p, m, threads, groups, idx;

  t = get_global_id(0);
  threads = get_global_size(0);

  m = threads; 
  groups = threads / m;

  for (p=0; p<passes; p++){
      
    idx = t * passes + p;

    map[idx].A = t + (t / m) * m;
    map[idx].B = map[idx].A + m;
    map[idx].W = (t % m) * groups;
    
    m = m >> 1;  
    groups = (m > 0) ? threads / m : 0;

  }
  
}

void radix2_dif_butterfly(const float2 A, const float2 B, const int k, const int N, __global const float2 *X, __global const float2 *Y)
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
 
  X->x = x.x;
  X->y = x.y;
  Y->x = z.x;
  Y->y = z.y;
}

__kernel void radix2_power_2_inplace_fft(__global struct fft_map *map, __global const float2 *in, const int N, const int passes)
{
  register int a, b, w, p, t, idx, threads;

  threads = get_global_size(0);
  t = get_global_id(0);
  
  for (p=0; p<passes; p++){
    
    idx = t * passes + p;

    a = map[idx].A;
    b = map[idx].B;
    w = map[idx].W;
    
    radix2_dif_butterfly(in[a], in[b], w, N, &in[a], &in[b]);

    barrier(CLK_GLOBAL_MEM_FENCE);
  }
  
}

__kernel void radix2_bit_reversal(__global const float2 *in, const int N)
{
  register int i;
  float2 x;

  i = get_global_id(0);

  

}

__kernel void uint8_re_to_float2(__global const unsigned char *in, __global const float2 *out)
{
  register int i;

  i = get_global_id(0);

  out[i].x = (float) in[i];
  out[i].y = 0.0;

}

#if 0
__kernel void uint8_cmplx_to_float2(__global const uint8_t *in, __global const float2 *out)
{
  register int i;

  i = get_global_id(0);

  out[i].x = (float) in[i];
  out[i].y = (float) in[i+1];

}
#endif
