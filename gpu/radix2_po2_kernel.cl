/**Author adam@ska.ac.za**/

//#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define THREAD get_global_id(1)*get_global_size(0)+get_global_id(0)

struct fft_map {
  int A;
  int B;
  int W;
};

struct bit_flip_map {
  int A;
  int B;
};

/*2d*/
__kernel void radix2_fft_setup(__global struct fft_map *map, const int passes, const int threads)
{
  int t, p, m, groups, idx;

  //t = get_global_id(0);
  t = THREAD; 

  m = threads; 
  groups = threads / m;

  for (p=0; p < passes && t < threads; p++){
      
    idx = t * passes + p;

    map[idx].A = t + (t / m) * m;
    map[idx].B = map[idx].A + m;
    map[idx].W = (t % m) * groups;
    
    m = m >> 1;  
    groups = (m > 0) ? threads / m : 0;

  }
  
}

#if 0 
void radix2_dif_butterfly(const float2 A, const float2 B, const int k, const int N, __global float2 *X, __global float2 *Y)
{
  float2 x, y, z, w; 
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
#endif

__kernel void radix2_power_2_inplace_fft(__global const struct fft_map *map, __global float2 *in, const int N, const int passes, const int threads)
{
  int a, b, k, p, t, m, groups, idx;
  float2 x, y, z, w;

  idx = 0;
  a = 0;
  b = 0;
  k = 0;

  //t = get_global_id(0);
  t = THREAD;
  m = threads;
  groups = threads / m;

  #pragma unroll
  for (p=0; p<passes && t < threads; p++){
    
    idx = t * passes + p;

#if 0
    a = map[idx].A;
    b = map[idx].B;
    k = map[idx].W;
#endif

    a = t + (t / m) * m;
    b = a + m;
    k = (t % m) * groups;

    //radix2_dif_butterfly(in[a], in[b], w, N, &in[a], &in[b]);

    x.x = in[a].x + in[b].x;
    x.y = in[a].y + in[b].y;

    y.x = in[a].x - in[b].x;
    y.y = in[a].y - in[b].y;

#if 1
    w.x = (float) native_cos(2.0f * M_PI_F * k / N);
    w.y = (float) (-1.0) * native_sin(2.0f * M_PI_F * k / N);
#endif

#if 0
    w.x = (float) cos(2.0 * M_PI_F * k / N);
    w.y = (float) (-1.0) * sin(2.0 * M_PI_F * k / N);
#endif

    z.x = (y.x * w.x) - (y.y * w.y);
    z.y = (y.y * w.x) + (y.x * w.y);
  
    in[a].x = x.x;
    in[a].y = x.y;
    in[b].x = z.x;
    in[b].y = z.y;

    //barrier(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_GLOBAL_MEM_FENCE);
    //mem_fence(CLK_GLOBAL_MEM_FENCE);

    m = m >> 1;  
    groups = (m > 0) ? threads / m : 0;

  }
  
}

__kernel void radix2_bit_flip_setup(__global struct bit_flip_map *flip, const int flips, const int N, const int passes)
{
  int i, r, in, count, have;
  
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

/*2d*/
__kernel void radix2_bit_flip(__global const struct bit_flip_map *flip, __global const float2 *in, const int flips)
{
  int i;
  float2 temp;

  //i = get_global_id(0);
  i = THREAD;
  
#if 0
  temp = in[flip[i].A];
  in[flip[i].A] = in[flip[i].B];
  in[flip[i].B] = temp;
#endif

#if 1
  if (i < flips){
    temp.x = in[flip[i].A].x;
    temp.y = in[flip[i].A].y;

    in[flip[i].A].x = in[flip[i].B].x;
    in[flip[i].A].y = in[flip[i].B].y;

    in[flip[i].B].x = temp.x;
    in[flip[i].B].y = temp.y;
  }
#endif
}

__kernel void uint8_re_to_float2(__global const unsigned char *in, __global const float2 *out, const int m)
{
  int i;

  i = THREAD; 

  if (i < m){
    out[i].x = (float) in[i];
    out[i].y = 0.0;
  }

}

__kernel void power_phase(__global const float2 *in, const int m)
{
  int i = THREAD; 
  float2 temp;

  float x,y;

  if (i < m) {

    x = in[i].x;
    y = in[i].y;

    //temp.x = hypot(in[i].x, in[i].y);
    temp.x = sqrt(x * x + y * y);
    temp.y = 0.0;//atan2(in[i].y, in[i].x);

    in[i].x = temp.x;
    in[i].y = temp.y;
  }

}

#define D        (4148.808)
#define DM       67
#define CF       1.5e9

__kernel void coherent_dedisperse(__global const float2 *in)
{
  int n = THREAD; 
  
  float freq = (float) n;
  float x,y;

  x = in[n].x;
  y = in[n].y;

  //const float p =  (2.0 * M_PI_F * DM * (freq * freq)) / ((freq+CF)*(CF*CF)) * D;
  const float p = 1.0 / ((freq+CF)*sqrt(CF)) * 2.0 * M_PI_F * DM * sqrt(freq) * D; 

  const float2 chirp = { 
    native_cos(p), 
    (-1) * native_sin(p) 
  };

  x = (float) (x * chirp.x) - (y * chirp.y);
  y = (float) (x * chirp.y) + (y * chirp.x);
}

__kernel void folder(__global const float2 *in, __global const float2 *fold_map, const int fold_id, const int N)
{
  int i = THREAD;
  int n = THREAD + N * fold_id;
   
  fold_map[n].x += in[i].x; 
  fold_map[n].y += in[i].y; 
}

__kernel void clmemset(__global const float2 *in)
{
  int n = THREAD;
  in[n].x = 0.0;
  in[n].y = 0.0;
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
