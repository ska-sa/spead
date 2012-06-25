#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sysexits.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/wait.h>


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

  key = ftok("/dev/null", 'A');
  if (key < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: ftok error: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }
  
  id = shmget(key, size, 0644 | IPC_CREAT);
  if (id < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: shmget error: %s\n", __func__, strerror(errno)); 
#endif
    return -1;
  }

  ptr = shmat(id, (void *) 0, 0);
  if (ptr == (void *) -1){
#ifdef DEBUG
    fprintf(stderr, "%s: shmat error %s\n", __func__, strerror(errno));
#endif
    if (shmctl(id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: shmctl error %s\n", __func__, strerror(errno)); 
#endif
      return -2;
    }
    return -1;
  }

  m_area = malloc(sizeof(struct shared_mem));
  if (m_area == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not allocate memory for shared memory store\n", __func__); 
#endif
    if (shmdt(ptr) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: shmdt error %s\n", __func__, strerror(errno)); 
#endif
    }
    if (shmctl(id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: shmctl error %s\n", __func__, strerror(errno)); 
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
  fprintf(stderr, "%s: shared memory or size [%ld] created shared_malloc now available\n", __func__, size);
#endif

  return 0;
}

void destroy_shared_mem()
{
  if (m_area) {
    if (shmdt(m_area->m_ptr) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: shmdt error %s\n", __func__, strerror(errno)); 
#endif
    }

    if (shmctl(m_area->m_id, IPC_RMID, NULL) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: shmctl error %s\n", __func__, strerror(errno)); 
#endif
    }
    
    free(m_area);
    m_area = NULL;
  }
}

void *shared_malloc(size_t size)
{
  struct shared_mem *m;
  void *ptr;

  m = m_area;
  if (m == NULL)
    return NULL;
 
  if (size < 0 && size + m->m_off < m->m_size)
    return NULL;

  ptr       = m->m_ptr + m->m_off;
  m->m_off  = m->m_off + size;
  
  return ptr;
}

#if 0
void shared_free(void *ptr)
{
  
}
#endif


#ifdef TEST_SHARED_MEM
#ifdef DEBUG

#define SIZE 5

struct test_mem{
  char m_test[100];
};

int main(int argc, char *argv[])
{
  struct test_mem *m[SIZE];
  int i,j;
  pid_t cpid;

  if (create_shared_mem(SIZE*sizeof(struct test_mem)) < 0){
    fprintf(stderr, "could not create shared mem\n");
    return 1;
  }

  for (i=0; i<SIZE; i++){
    m[i] = shared_malloc(sizeof(struct test_mem));
    if (m[i] == NULL){
      fprintf(stderr, "shared_malloc fail\n");
      destroy_shared_mem();
      return 1;
    }
  }
  
#if 0

#endif

  for (j=0; j<2; j++){
    cpid = fork();

    if (cpid < 0){
      fprintf(stderr, "fork error %s\n", strerror(errno));
      return 1;
    }

    if (!cpid){
      for (i=0; i<SIZE; i++){
        fprintf(stderr, "child [%d] writing\n", getpid());
        snprintf(m[i]->m_test, sizeof(m[i]->m_test), "i [%d] am block number %d\n", getpid(), i);
      }

      exit(EX_OK);
    }

    fprintf(stderr, "PARENT [%d] forked child [%d]\n", getpid(), cpid);
  }


  wait(NULL);

  for (i=0; i<SIZE; i++){
    fprintf(stderr, "parent: %s", m[i]->m_test);
  }

  destroy_shared_mem(); 
  return 0;
}
#endif
#endif
