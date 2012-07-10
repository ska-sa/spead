#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
#include <stdint.h>

#include <asm-generic/mman.h>
#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/wait.h>

#include "mutex.h"

static pid_t mypid;

#if 0 
#define IRQ_DISABLED 0
#define IRQ_ENABLED  1

static inline unsigned long native_save_fl(void)
{
	unsigned long flags;

	/*
	 * "=rm" is safe here, because "pop" adjusts the stack before
	 * it evaluates its effective address -- this is part of the
	 * documented behavior of the "pop" instruction.
	 */
	asm volatile("# __raw_save_flags\n\t"
		     "pushf ; pop %0"
		     : "=rm" (flags)
		     : /* no input */
		     : "memory");

	return flags;
}

static inline void native_restore_fl(unsigned long flags)
{
	asm volatile("push %0 ; popf"
		     : /* no output */
		     :"g" (flags)
		     :"memory", "cc");
}

static inline unsigned long arch_local_irq_save(void)
{
	unsigned long flags;
	flags = native_save_fl();
	native_restore_fl(IRQ_DISABLED);
	return flags;
}

static inline void arch_local_irq_restore(unsigned long flags)
{
  native_restore_fl(flags);
}

#define irq_restore(flags)			\
	do {						\
		arch_local_irq_restore(flags);		\
	} while (0)
#define irq_save(flags)			\
	do {						\
		flags = arch_local_irq_save();	\
	} while (0)
#endif


void lock_mutex(mutex *m)
{
  mutex c;
  volatile int i;
      
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
  volatile int i;
  
  if (*m == 2){
    *m = 0;
  } else if (xchg(m, 0) == 1){
    return;
  }
  
  for (i=0; i<200; i++){
    if (*m){
      if (cmpxchg(m, 1, 2))
        return;
    }
    cpu_relax();
  }
    
  syscall(SYS_futex, m, FUTEX_WAKE, 1, NULL, NULL, 0);
  
  return;
}

int main(int argc, char *argv[])
{
#define CHILD 10
  int i, j;
  pid_t cpid;
  unsigned long *v;
  mutex *key;

  fprintf(stderr, "sizeof (unsigned long) %ldbytes\nsizeof (mutex) %ldbytes\n", sizeof(unsigned long), sizeof(mutex));

  v = mmap(NULL, sizeof(unsigned long) + sizeof(mutex) , PROT_READ | PROT_WRITE | PROT_SEM | PROT_EXEC, MAP_ANONYMOUS | MAP_SHARED, (-1), 0);
  if (v == NULL)
    return 1;

  key = (mutex *)(v + sizeof(unsigned long));

  *v   = 0;
  *key = 0;

#if 0
  *v = cmpxchg(key, 0, 1);

  fprintf(stderr, "cmpxchgq return %ld key %ld\n", *v, *key);

  *v = xchg(key, 5);

  fprintf(stderr, "xchgq return %ld key %ld\n", *v, *key);

  *v = atomic_dec(key);

  fprintf(stderr, "atomic_dec return %ld key %ld\n", *v, *key);
#endif

  

#if 1
  for (j=0; j<CHILD; j++){

    cpid = fork();

    if (cpid < 0){
      fprintf(stderr, "fork error %s\n", strerror(errno));
      return 1;
    }

    if (!cpid){
      mypid = getpid();

      /*THIS IS A CHILD*/
      fprintf(stderr, "CHILD [%d] writing\n", mypid);

      //pause();

      for (i=0; i<1000; i++){
        lock_mutex(key);
        (*v)++;
        unlock_mutex(key);
        usleep(1);
      }

      exit(EX_OK);
    }

    //fprintf(stderr, "PARENT [%d] forked child [%d]\n", getpid(), cpid);
  }

  i=0;
  do {
    fprintf(stderr, "parent collected child [%d]\n", wait(NULL));
    //wait(NULL);
  } while(i++ < CHILD);

  fprintf(stderr, "PARENT VALUE [%ld]\n", *v);

#endif
  munmap(v, sizeof(unsigned long) + sizeof(mutex));
  return 0;
}

