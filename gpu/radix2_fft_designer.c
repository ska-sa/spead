#include <stdio.h>
#include <math.h>

int is_power_of_2(int x)
{
  return (x != 0) ? ((x & (x-1)) == 0 ? 0 : -1) : -1;
}

int power_of_2(int x)
{
  int p=0;
  while (x > 1) {
    x = x >> 1;
    p++;
  }
  return p;
}

int main(int argc, char *argv[])
{
  int t, p, a, b, groups, m, w;
  int N = atoi(argv[1]);
  
  int passes, threads;

  if (is_power_of_2(N) < 0){
    fprintf(stderr, "%d not a power of two\n", N);
    return 1;
  }
   
  passes = power_of_2(N);

  if (passes == 0){
    fprintf(stderr, "need a minimum of 2 data points\n");
    return 1;
  }

  threads = N >> 1;
  m = threads;
  groups = threads / m;

  fprintf(stderr, "%d passes needed by %d threads\n", passes, threads);
  
#if 0
  for (p=0; p<passes; p++){
    for (t=0; t<threads; t++){
      
      a = t + (t / m) * m;
      b = a + m; 
      w = (t % m) * groups;

      fprintf(stderr, "groups [%d] of m[%d] thread[%d] pass[%d] A[%d] B[%d] W[%d]\n", groups, m, t, p, a, b, w); 
      if ((t+1) % m == 0)
        fprintf(stderr, "\n");

    }
  
    m = m >> 1;
    groups = (m > 0) ? threads / m : 0;

    fprintf(stderr, "----------------\n");

  }
#endif
  int need = 0, same = 0;
#if 1
  for (t=0; t < N; t++){
    
    int r = t, in=0;
    int count=0;

    while (count < passes){
      count++;
      in = in << 1;
      in = in | (r & 0x1);
      r = r >> 1;
    }

#if 0
    fprintf(stderr, "bit-reversal of [%d] is [%d]", t, in);
    if (t < in){
      fprintf(stderr, "\tdo swap use %d", need);
      need++;
    } else if (t == in) {
      fprintf(stderr, "\t====");
      same++;
    } else {
      fprintf(stderr, "\t😸  ");
    }

    fprintf(stderr, "\n");
#endif
#if 1
    if (t < in){
      fprintf(stderr, "%d\t%d-%d\n", need, t, in);
      need++;
    }
#endif
  }

  fprintf(stderr, "need count %d\n", need);
#endif

  int diff  = (passes % 2) ? /*odd*/ ((passes+1) >> 1) - 1 : /*even*/ (passes >> 1) - 1;
  int flips = ((N >> 1) - (1 << diff));
  
  fprintf(stderr, "calculated flips needed: diff %d flips %d same %d\n", diff, flips, same);
#if 0
  for (t=0; t<flips; t++)
  {
    
  }
#endif

  return 0;
}
