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
  
  if (argc != 2){
    fprintf(stderr, "need a N\n");
    return 1;
  }

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
  


  
  return 0;
}


