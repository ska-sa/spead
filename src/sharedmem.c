#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sysexits.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/wait.h>


#include "sharedmem.h"

#define KEYPATH     "/dev/null"
#define MAX_RETRIES 10

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

  key = ftok(KEYPATH, 'A');
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

  memset(ptr, 0, size);

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
 
  if (size < 0 || (size + m->m_off) > m->m_size){
#ifdef DEBUG
    fprintf(stderr, "%s: FAIL shared_malloc size req [%ld] mem stats msize [%ld] m_off [%ld]\n", __func__, size, m->m_size, m->m_off); 
#endif
    return NULL;
  }

  ptr       = m->m_ptr + m->m_off;
  m->m_off  = m->m_off + size;
  
#if DEBUG>1
  fprintf(stderr, "%s: allocated [%ld] from sharedmem\n", __func__, size);
#endif

  return ptr;
}

#if 0
void shared_free(void *ptr)
{
  
}
#endif

int create_sem()
{
  int semid, i, ready;
  union semun arg;
  struct semid_ds buf;
  struct sembuf sb;
  key_t key;

  key = ftok(KEYPATH, 'B');
  if (key < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: ftok error: %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  semid = semget(key, 1, IPC_CREAT | IPC_EXCL | 0666);
  
  if (semid < 0){ 
    switch(errno){
      case EEXIST: /*semaphore is already created*/
        ready = 1;
        semid = semget(key, 1, 0);
        if (semid < 0){
#ifdef DEBUG
          fprintf(stderr, "%s: error semget %s\n", __func__, strerror(errno));
#endif
          return -1;
        }

        arg.buf = &buf;
        for (i=0; i<MAX_RETRIES && !ready; i++){
          semctl(semid, 0, IPC_STAT, arg);
          if (arg.buf->sem_otime == 0) {
            ready = 1;
          } else {
            sleep(1);
          }
        }

        if (!ready){
          errno = ETIME;
#ifdef DEBUG
          fprintf(stderr, "%s: error waiting %s\n", __func__, strerror(errno));
#endif
          return -1;
        }

        return semid;

      default:
#ifdef DEBUG
        fprintf(stderr, "%s: error semget %s\n", __func__, strerror(errno));
#endif
        return -1;
    }
  }

  /*set semaphore to unlocked*/
  sb.sem_num = 0;
  sb.sem_op  = 1;
  sb.sem_flg = 0;
  arg.val    = 1;

  if (semop(semid, &sb, 1) < 0) {
#ifdef DEBUG
    fprintf(stderr, "%s: error semop %s\n", __func__, strerror(errno));
#endif
    semctl(semid, 0, IPC_RMID);
    return -1;
  }

  return semid; 
}

/*test and lock semaphore*/
int lock_sem(int semid)
{
  struct sembuf sb;
  
  if (semid < 0)
    return -1;

  sb.sem_num =   0;
  sb.sem_op  = (-1);
  sb.sem_flg = SEM_UNDO;
  
  if (semop(semid, &sb, 1) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error semop %s\n", __func__, strerror(errno));
#endif
    return -1;
  }

  return 0;
}

int unlock_sem(int semid)
{
  struct sembuf sb;

  if (semid < 0)
    return -1;

  sb.sem_num = 0;
  sb.sem_op  = 1;
  sb.sem_flg = SEM_UNDO;

  if (semop(semid, &sb, 1) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: error semop %s\n", __func__, strerror(errno));
#endif
    return -1;
  }
  
  return 0;
}

void destroy_sem(int semid)
{
  if (semid > 0){
    
    if (semctl(semid, 0, IPC_RMID) < 0) {
      fprintf(stderr, "%s: error semctl %s\n", __func__, strerror(errno));
    }

  }
}


#ifdef TEST_SHARED_MEM
#ifdef DEBUG

#define SIZE      50
#define CHILD     40

struct test_mem{
  /*char m_test[100];
  */
  int v;
};

int main(int argc, char *argv[])
{
  struct test_mem *m[SIZE];
  int semid, i, j;
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
    bzero(m[i], sizeof(struct test_mem));
  }
  
 semid = create_sem();
 //semid=0;
  if (semid < 0){
    fprintf(stderr, "could not create semaphore for shared mem\n"); 
    destroy_shared_mem();
    return 1;
  }

  for (j=0; j<CHILD; j++){

    cpid = fork();

    if (cpid < 0){
      fprintf(stderr, "fork error %s\n", strerror(errno));
      return 1;
    }

    if (!cpid){
      /*THIS IS A CHILD*/

      for (i=0; i<SIZE; i++){
        //if (!strlen(m[i]->m_test)){
          fprintf(stderr, "CHILD [%d] writing %d\n", getpid(), i);
          lock_sem(semid);
#if 0
          snprintf(m[i]->m_test, sizeof(m[i]->m_test), "i [%d] am block number %d\n", getpid(), i);
#endif
          m[i]->v++;
          unlock_sem(semid);
        //}
      }

      exit(EX_OK);
    }

    fprintf(stderr, "PARENT [%d] forked child [%d]\n", getpid(), cpid);
  }

  i=0;
  do {
    fprintf(stderr, "parent collected child [%d]\n", wait(NULL));
  } while(i++ < CHILD);

  for (i=0; i<SIZE; i++){
    //fprintf(stderr, "parent: %s", m[i]->m_test);
    fprintf(stderr, "parent: %d\n", m[i]->v);
  }

  destroy_sem(semid);
  destroy_shared_mem(); 
  return 0;
}
#endif
#endif
