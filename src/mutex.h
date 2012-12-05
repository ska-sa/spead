/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#ifndef MUTEX_H
#define MUTEX_H

#define cpu_relax() \
  __asm__ __volatile__ ( "pause\n" : : : "memory")

typedef int mutex;

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

