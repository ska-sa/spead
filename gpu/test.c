#include <stdio.h>
#include <math.h>

int is_power_of_2(int x)
{
  return (x != 0) ? ((x & (x-1)) == 0 ? 0 : -1) : -1;
}

int main(int argc, char *argv[])
{
  int i, a, b, idx, m;
  int N = atoi(argv[1]);
  
  float passes;

  if (is_power_of_2(N) < 0){
    fprintf(stderr, "%d not a power of two\n", N);
    return 1;
  }
   
  fprintf(stderr, "%d is power of two\n", N);
  return 0;

  for (i=0; i<passes; i++){
    
    a = idx;
    b = a+m;
      
    fprintf(stderr, "pass %d m[%d] A[%d] B[%d]\n", i, m, a, b);
    
    m = m >> 1;

  }

  
  return 0;
}
