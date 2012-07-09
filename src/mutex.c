#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
#include <stdint.h>

#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/wait.h>

#define LOCK_PREFIX \
    ".section .smp_lock,\"a\"\n" \
    ".balign 4\n" \
    ".long 671f - .\n" \
    ".previous\n" \
    "671:"

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

static inline unsigned long cmpxchg(volatile void *ptr, volatile unsigned long old, volatile unsigned long new)
{
  volatile unsigned long ret, flags;
  
  //irq_save(flags);

  asm volatile(LOCK_PREFIX "cmpxchgq %2,%1"
              : "=a" (ret), "+m" (*(unsigned long *)ptr)
              : "r" (new), "0" (old)
              : "memory");

  //irq_restore(flags);
   
  return ret;
}

static inline unsigned long xchg(volatile void *ptr, volatile unsigned long x)
{
  volatile unsigned long flags;
  //irq_save(flags);
  asm volatile("xchgq %0,%1"
              :"=r" (x), "+m" (*(unsigned long*)ptr)
              :"0" (x)
              :"memory");

  //irq_restore(flags);
  return x;
}

static inline unsigned long atomic_dec(volatile void *ptr)
{
  volatile unsigned long ret, flags;

  //irq_save(flags);
  ret = *(unsigned long*)ptr;
  asm volatile(LOCK_PREFIX "decl %0"
              : "+m" (*(unsigned long*)ptr));

  //irq_restore(flags);
  return ret;
}

void lock_mutex(unsigned long *key)
{
  volatile unsigned long c;
  c = 0;
  
  if ((c = cmpxchg(key, 0, 1)) != 0){

    if (c != 2){
      c = xchg(key, 2);
    }

    while (c != 0){
      syscall(SYS_futex, key, FUTEX_WAIT, 2, NULL, NULL, 0);
      c = xchg(key, 2);
    }
  } 

}

void unlock_mutex(unsigned long *key)
{
  if (atomic_dec(key) != 1){
    *(key) = 0;
    syscall(SYS_futex, key, FUTEX_WAKE, 1, NULL, NULL, 0);
  } 
}

int main(int argc, char *argv[])
{
#define CHILD 3
  int i, j;
  pid_t cpid;
  unsigned long *v, *key;

  fprintf(stderr, "sizeof (unsigned long) %ldbytes\nsizeof (uint64_t) %ldbytes\n", sizeof(unsigned long), sizeof(uint64_t));

  v = mmap(NULL, sizeof(unsigned long)*2, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, (-1), 0);
  if (v == NULL)
    return 1;

  key = v + sizeof(unsigned long);

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
  munmap(v, sizeof(unsigned long)*2);
  return 0;
}

