#include <stdio.h>
#include <math.h>

int is_power_of_4(int x)
{
  //int a = (x == 0) ? -1 : ((x & 0x55555555) & 0x00000001 == 0 ? 0 : -1);
  int a = (x & 0x55555555) & 0x00000001;

  fprintf(stderr,"a=%d\n", a);

  return a;
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
  int t, p, a, b, groups, m, w, N;
  
  int passes, threads;
  
  if (argc != 2){
    fprintf(stderr, "need a N\n");
    return 1;
  }

  N = atoi(argv[1]);

  if (is_power_of_4(N) < 0){
    fprintf(stderr, "%d not a power of four\n", N);
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
  


  
  return 0;
}


