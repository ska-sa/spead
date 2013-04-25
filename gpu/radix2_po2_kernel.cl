/**Author adam@ska.ac.za**/

//#pragma OPENCL EXTENSION cl_khr_fp64: enable


struct fft_map {
  int A;
  int B;
  int W;
};

struct bit_flip_map {
  int A;
  int B;
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
  register float2 x, y, z, w; 
#if 1
  x.x = A.x + B.x;
  x.y = A.y + B.y;
  
  y.x = A.x - B.x;
  y.y = A.y - B.y;
#endif

  w.x = (float) cos(2.0 * M_PI_F * k / N);
  w.y = (float) (-1.0) * sin(2.0 * M_PI_F * k / N);

#if 1
  z.x = (y.x * w.x) - (y.y * w.y);
  z.y = (y.y * w.x) + (y.x * w.y);
#endif

#if 0
  x = A + B;
  y = A - B;
  z = y * w;
#endif

  X->x = x.x;
  X->y = x.y;
  Y->x = z.x;
  Y->y = z.y;
}

__kernel void radix2_power_2_inplace_fft(__global const struct fft_map *map, __global const float2 *in, const int N, const int passes)
{
  register int a, b, w, p, t, idx, threads;

  idx = 0;
  a = 0;
  b = 0;
  w = 0;

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

__kernel void radix2_bit_flip_setup(__global struct bit_flip_map *flip, const int flips, const int N, const int passes)
{
  register int i, r, in, count, have;
  
  have = 0;

  for (i=0; i<N || have < flips; i++){
    
    r = i;
    in = 0;
    count = 0;

    while (count < passes){
      count++;
      in = in << 1;
      in = in | (r & 0x1);
      r = r >> 1;
    }

    if (i < in){
      flip[have].A = i;
      flip[have].B = in;
      have++;
    }

    if (have == flips){
      //i=N;
      break;
    }
  }

}

__kernel void radix2_bit_flip(__global const struct bit_flip_map *flip, __global const float2 *in, const int flips)
{
  register int i;
  register float2 temp;

  i = get_global_id(0);
  
#if 0
  temp = in[flip[i].A];
  in[flip[i].A] = in[flip[i].B];
  in[flip[i].B] = temp;
#endif

#if 1
  temp.x = in[flip[i].A].x;
  temp.y = in[flip[i].A].y;

  in[flip[i].A].x = in[flip[i].B].x;
  in[flip[i].A].y = in[flip[i].B].y;

  in[flip[i].B].x = temp.x;
  in[flip[i].B].y = temp.y;
#endif
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
