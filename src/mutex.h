#ifndef MUTEX_H
#define MUTEX_H

#define cpu_relax() \
  __asm__ __volatile__ ( "pause\n" : : : "memory")

#define LOCK_PREFIX \
    ".section .smp_lock,\"a\"\n" \
    ".balign 4\n" \
    ".long 671f - .\n" \
    ".previous\n" \
    "671:"

typedef volatile int mutex;

static inline mutex cmpxchg(mutex *ptr, mutex old, mutex new)
{
  mutex ret;
  
  asm volatile(LOCK_PREFIX "cmpxchgl %2,%1"
              : "=a" (ret), "+m" (*(mutex *)ptr)
              : "r" (new), "0" (old)
              : "memory");
   
  return ret;
}

static inline mutex xchg(mutex *ptr, mutex x)
{
  asm volatile("xchgl %0,%1"
              :"=r" (x), "+m" (*(mutex*)ptr)
              :"0" (x)
              :"memory");

  return x;
}

static inline mutex atomic_dec(mutex *ptr)
{
  mutex ret;

  ret = *(mutex*)ptr;
  asm volatile(LOCK_PREFIX "decl %0"
              : "+m" (*(mutex*)ptr));

  return ret;
}


#endif

