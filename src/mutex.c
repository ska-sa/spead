#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>

#include <linux/futex.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/wait.h>

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


static inline unsigned long cmpxchg(volatile void *ptr, unsigned long old, unsigned long new)
{
	unsigned long flags, prev;

	irq_save(flags);

	prev = *(unsigned long *)ptr;
	if (prev == old){
	  *(unsigned long *)ptr = (unsigned long)new;
  }

	irq_restore(flags);

  return prev;
}

static inline unsigned long xchg(volatile void *ptr, unsigned long x)
{
  unsigned long ret, flags;

  irq_save(flags);
  
  ret = *(unsigned long *) ptr;
  *(unsigned long *) ptr = x;

  irq_restore(flags);
  return ret;
}

static inline unsigned long atomic_dec(volatile void *ptr)
{
  unsigned long flags;

  irq_save(flags);

  *(unsigned long *)ptr -= 1;
  
  irq_restore(flags);

  return *(unsigned long *)ptr;
}

void lock_mutex(unsigned long *key)
{
  int c;
  
  if ((c = cmpxchg(key, 0, 1)) != 0){

#if 0
def DEBUG
    fprintf(stderr, "%s: cmpxchg 0 1 rtn c prev = %d\n", __func__, c);
#endif
    
    if (c != 2)
      c = xchg(key, 2);

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
#define CHILD 2
  int i, j;
  pid_t cpid;
  unsigned long *v, *key;

  v = mmap(NULL, sizeof(unsigned long)*2, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_SHARED, (-1), 0);
  if (v == NULL)
    return 1;


  key = v + sizeof(unsigned long);

  *v   = 0;
  *key = 0;


  for (j=0; j<CHILD; j++){

    cpid = fork();

    if (cpid < 0){
      fprintf(stderr, "fork error %s\n", strerror(errno));
      return 1;
    }

    if (!cpid){
      /*THIS IS A CHILD*/
      fprintf(stderr, "CHILD [%d] writing\n", getpid());

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

  munmap(v, sizeof(unsigned long)*2);

  return 0;
}

