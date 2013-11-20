/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#ifndef MUTEX_H
#define MUTEX_H


typedef int mutex;

#ifndef ARCH

#define cpu_relax() \
  __asm__ __volatile__ ( "pause\n" : : : "memory")

static inline mutex cmpxchg(mutex *ptr, mutex old, mutex n)
{
  mutex ret;
  asm volatile("lock\n" "cmpxchgl %2,%1\n"
              : "=a" (ret), "+m" (*(mutex *)ptr)
              : "r" (n), "0" (old)
              : "memory");
   
  return ret;
}

static inline mutex xchg(mutex *ptr, mutex x)
{
  asm volatile("lock\n" "xchgl %0,%1\n"
              :"=r" (x), "+m" (*(mutex*)ptr)
              :"0" (x)
              :"memory");
  return x;
}

#else

#define cpu_relax() \
  __asm__ __volatile__ ( "" : : : "memory")

static inline mutex cmpxchg(mutex *ptr, mutex old, mutex n)
{
  mutex ret;
  
  asm volatile ( "1: lwarx  %0,0,%2\n\
                     cmpw   0,%0,%3\n\
                     bne-   2f\n\
                     stwcx. %4,0,%2\n\
                     bne-   1b\n\
                  2:"
                  : "=&r" (ret), "+m" (*(mutex *)ptr)
                  : "r" (ptr), "r" (old), "r" (n)
                  : "cc", "memory");

  return ret;
}

static inline mutex xchg(mutex *ptr, mutex x)
{
  mutex ret;

  asm volatile ( "1: lwarx  %0,0,%2\n\
                     stwcx. %3,0,%2\n\
                     bne-   1b"
                  : "=&r" (ret), "+m" (*(mutex *)ptr)
                  : "r" (ptr), "r" (x)
                  : "cc", "memory");
  
  return ;
}

#endif

void lock_mutex(mutex *m);
void unlock_mutex(mutex *m);

#if 0
static inline mutex atomic_dec(mutex *ptr)
{
  mutex ret;

  ret = *(mutex*)ptr;
  asm volatile(LOCK_PREFIX "decl %0"
              : "+m" (*(mutex*)ptr));

  return ret;
}
#endif

#endif

