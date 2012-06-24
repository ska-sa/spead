#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/ipc.h>


#include "sharedmem.h"

static struct shared_mem *m_area = NULL;


int create_shared_mem(uint64_t size)
{
  key_t key;
  int id;
  void *ptr;

  if (m_area != NULL) {
#ifdef DEBUG
    fprintf(stderr, "%s: a shared memory segment is already assigned\n", __func__); 
#endif
    return -1;
  }

  if (size < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: a shared memory segment must have a positive size\n", __func__); 
#endif
    return -1;
  }

  key = ftok("/dev/zero", 'A');
  if (key < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }
  
  id = shmget(key, size, 0644 | IPC_CREAT);
  if (id < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error: %s\n", __func__, strerror(errno)); 
#endif
    return -1;
  }

  ptr = shmat(key, (void *) 0, 0);
  if (ptr == (void *) -1){
#ifdef DEBUG
    fprintf(stderr, "%s: error %s\n", __func__, strerror(errno));
#endif
    if (shmctl(id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error %s\n", __func__, strerror(errno)); 
#endif
      return -2;
    }
    return -1;
  }

  m_area = malloc(sizeof(struct m_area));
  if (m_area == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not allocate memory for shared memory store\n", __func__); 
#endif
    if (shmdt(ptr) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error %s\n", __func__, strerror(errno)); 
#endif
    }
    if (shmctl(id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error %s\n", __func__, strerror(errno)); 
#endif
      return -2;
    }
    return -1;
  }

  m_area->m_key  = key;
  m_area->m_id   = id;
  m_area->m_size = size;
  m_area->m_off  = 0;
  m_area->m_ptr  = ptr;

#ifdef DEBUG
  fprintf(stderr, "%s: shared memory or size [%ld] created shared_malloc now available\n", __func__);
#endif

  return 0;
}

void destroy_shared_mem()
{
  if (m_area) {
    if (shmdt(m_area->m_ptr) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error %s\n", __func__, strerror(errno)); 
#endif
    }
    if (shmctl(m_area->m_id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error %s\n", __func__, strerror(errno)); 
#endif
    }
  }
}

void *shared_malloc(size_t size)
{
  void *ptr;
  


  return ptr;
}

void shared_free(void *ptr)
{
  
  
}

