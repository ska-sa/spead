#ifndef SHAREDMEM_H
#define SHAREDMEM_H



struct shared_mem {
  key_t m_key;
  int m_id;
  uint64_t m_size;
  uint64_t m_off;
  void *m_ptr;
};

int create_shared_mem(uint64_t size);

void *shared_malloc(size_t *size);
void shared_free(void *ptr);

#endif
