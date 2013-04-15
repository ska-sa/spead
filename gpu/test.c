#include <stdio.h>
#include <math.h>

int is_power_of_2(int x)
{
  return (x != 0) ? ((x & (x-1)) == 0 ? 0 : -1) : -1;
}

int power_of_2(int x)
{
  int p=0;
  do {
    x = x >> 1;
    p++;
  } while(x > 0);
  return (p-1);
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
  threads = N >> 1;
  m = threads;
  groups = threads / m;

  fprintf(stderr, "%d passes needed\n", passes);
  

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
  
  return 0;
}
