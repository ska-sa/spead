#ifndef SHAREDMEM_H
#define SHAREDMEM_H

#include <stdint.h>

struct shared_mem {
  key_t m_key;
  int m_id;
  uint64_t m_size;
  uint64_t m_off;
  void *m_ptr;
};

#if 0
union semun {
  int val;
  struct semid_ds *buf;
  ushort *array;
};
#endif

int create_shared_mem(uint64_t size);
void destroy_shared_mem();
void *shared_malloc(size_t size);

#if 0
int create_sem(int c);
int lock_sem(int semid);
int unlock_sem(int semid);

void destroy_sem(int semid);
#endif

#if 0
void shared_free(void *ptr);
#endif

#endif
