#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
#include <stdint.h>
#include <limits.h>

#include <asm-generic/mman.h>
#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "mutex.h"

void lock_mutex(mutex *m)
{
  int c, i;
      
  for (i=0; i<100; i++){
    c = cmpxchg(m, 0, 1);
    if (!c)
      return;
    cpu_relax();
  }
  
  if (c == 1)
    c = xchg(m, 2);

  while (c){
    syscall(SYS_futex, m, FUTEX_WAIT, 2, NULL, NULL, 0);
    c = xchg(m, 2);
  }
  
  return;
}

void unlock_mutex(mutex *m)
{
  int i;
  
  if ((*m) == 2){                 
    (*m) = 0;                    
  } else if (xchg(m, 0) == 1){ 
    return;
  }
  
  for (i=0; i<200; i++){
    if ((*m)){
      if (cmpxchg(m, 1, 2)){
        return;
      }
    }
    cpu_relax();
  }
    
  syscall(SYS_futex, m, FUTEX_WAKE, 1, NULL, NULL, 0);
  
  return;
}

#ifdef TEST_MUTEX
int main(int argc, char *argv[])
{
#define CHILD 30
  int i, j;
  pid_t cpid;
  unsigned long *v;
  mutex *key;

  key = mmap(NULL, sizeof(unsigned long) + sizeof(mutex), PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, (-1), 0);
  if (key == NULL)
    return 1;
  
  bzero((void*)key, sizeof(unsigned long) + sizeof(mutex));

  v = (unsigned long *)(key + sizeof(mutex));

  *v   = 0;
  *key = 0;

#if 0
  *v = cmpxchg(key, 0, 1);
  fprintf(stderr, "cmpxchgq return %ld key %ld\n", *v, *key);
  *v = cmpxchg(key, 1, 2);
  fprintf(stderr, "cmpxchgq return %ld key %ld\n", *v, *key);
  *v = cmpxchg(key, 0, 1);
  fprintf(stderr, "cmpxchgq return %ld key %ld\n", *v, *key);
  *v = cmpxchg(key, 2, 0);
  fprintf(stderr, "cmpxchgq return %ld key %ld\n", *v, *key);

  *v = xchg(key, 5);
  fprintf(stderr, "xchgq return %ld key %ld\n", *v, *key);
#if 0
  *v = atomic_dec(key);
  fprintf(stderr, "atomic_dec return %ld key %ld\n", *v, *key);
#endif
#endif

#if 1
  for (j=0; j<CHILD; j++){

    cpid = fork();

    if (cpid < 0){
      fprintf(stderr, "fork error %s\n", strerror(errno));
      return 1;
    }

    if (!cpid){

      /*THIS IS A CHILD*/
      fprintf(stderr, "CHILD [%d] writing\n", getpid());


      for (i=0; i<100000; i++){
        lock_mutex(key);
        int temp = *v;
        temp++;
        *v = temp;
        unlock_mutex(key);
        usleep(1);
      }

      exit(EX_OK);
    }

  }

  i=0;
  do {
    fprintf(stderr, "parent collected child [%d]\n", wait(NULL));
  } while(i++ < CHILD);

  fprintf(stderr, "PARENT VALUE [%ld]\n", *v);

#endif
  munmap((void*) key, sizeof(unsigned long) + sizeof(mutex));
  return 0;
}
#endif
