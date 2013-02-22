/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sysexits.h>

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/wait.h>
#include <sys/mman.h>

#include "spead_api.h"
#include "server.h"
#include "avltree.h"

static struct shared_mem *m_area = NULL;

int compare_shared_mem_regions(const void *v1, const void *v2)
{
  if (*(uint64_t *)v1 < *(uint64_t *)v2)
    return -1;
  else if (*(uint64_t *)v1 > *(uint64_t *)v2)
    return 1;
  return 0;
}

int compare_shared_mem_size(const void *v1, const void *v2)
{
  if (*(size_t *) v1 < *(uint64_t *)v2)
    return -1;
  else if (*(size_t *) v1 > *(uint64_t *)v2)
    return 1;
  return 0;
}

struct shared_mem_region *create_shared_mem_region(uint64_t mid, uint64_t size)
{
  struct shared_mem_region *r;
  void *ptr;

  if (size <= 0)
    return NULL;

  ptr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (ptr == MAP_FAILED){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error %s\n", __func__, strerror(errno));
#endif
    return NULL;
  }
  
  r = mmap(NULL, sizeof(struct shared_mem_region), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (r == MAP_FAILED){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error %s\n", __func__, strerror(errno));
#endif
    munmap(ptr, size);
    return NULL;
  }
  
  r->m_id   = mid;
  r->m_size = size;
  r->m_off  = 0;
  r->m_ptr  = ptr;
  
  return r;
}

void destroy_shared_mem_region(void *data)
{ 
  struct shared_mem_region *r = data;
  if (r){
    if (r->m_ptr)
      munmap(r->m_ptr, r->m_size);
    munmap(r, sizeof(struct shared_mem_region));
  }
}

int add_shared_mem_region(struct shared_mem *m, uint64_t size)
{
  struct shared_mem_region *r;

  if (m == NULL || size <= 0)
    return -1;

  r = create_shared_mem_region(m->m_next_id, size);
  if (r == NULL)
    return -1;

  if (store_named_node_avltree(m->m_tree, &(r->m_id), r) < 0){
    destroy_shared_mem_region(r);
    return -1;
  }

  m->m_current = r;
  m->m_next_id++;

  return 0;
}

int create_shared_mem()
{
  struct shared_mem *m;
  struct avl_tree *t, *f;

  if (m_area != NULL)
    return -1;

  m = mmap(NULL, sizeof(struct shared_mem), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (m == MAP_FAILED){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error %s\n", __func__, strerror(errno));
#endif
    return -1;
  }
  
  m_area = m;
  
  /*GENISIS BLOCK*/
  m->m_current = create_shared_mem_region(0, SHARED_MEM_REGION_SIZE);
  if (m->m_current == NULL){
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }

  t = create_avltree(&compare_shared_mem_regions);
  if (t == NULL){
    destroy_shared_mem_region(m->m_current);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }

  f = create_avltree(&compare_shared_mem_size);
  if (f == NULL){
    destroy_shared_mem_region(m->m_current);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }

  if (store_named_node_avltree(t, &(m->m_current->m_id), m->m_current) < 0){
    destroy_shared_mem_region(m->m_current);
    destroy_avltree(t, NULL);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }

  m->m_tree    = t;
  m->m_free    = f;
  m->m_next_id = 1;

  return 0;
}

void destroy_shared_mem()
{
  if (m_area) {
#if 0
    //destroy_avltree(m_area->m_free, NULL);
    //destroy_avltree(m_area->m_tree, &destroy_shared_mem_region);
#endif

#ifdef DEBUG
    fprintf(stderr, "%s: mem regions %ld free nodes %ld\n", __func__, m_area->m_tree->t_ncount, m_area->m_free->t_ncount);
#endif
    

    munmap(m_area, sizeof(struct shared_mem));
    m_area = NULL;
  }
}

struct shared_mem_region *get_current_shared_mem_region(struct shared_mem *m)
{
  return (m)?m->m_current:NULL;
}

void *shared_malloc(size_t size)
{
  struct shared_mem_region *r;
  struct shared_mem_size   *s;
  struct shared_mem_free   *f;
  void *ptr;

  r = get_current_shared_mem_region(m_area);
  if (r == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: shared memory region is NULL\n",__func__);
#endif
    return NULL;
  }

  if (size < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: a shared memory segment must have a positive size\n", __func__); 
#endif
    return NULL;
  }

  s = find_name_node_avltree(m_area->m_free, &size);
  if (s != NULL) {
#ifdef DEBUG
    fprintf(stderr, "%s: found a free list for size %ld\n", __func__, size);
#endif

    if (s->s_top){
      
      f = s->s_top;
      ptr = s->s_top;

      s->s_top = f->f_next;

#ifdef DEBUG
      fprintf(stderr, "%s: ptr (%p)\n", __func__, ptr);
#endif

      return ptr;
    }

#ifdef DEBUG
    fprintf(stderr, "%s: no nodes in list\n", __func__);
#endif

  }
 
  if ((size + r->m_off) > r->m_size){
#ifdef DEBUG
    fprintf(stderr, "%s: WARN shared_malloc size req [%ld] mem stats msize [%ld] m_off [%ld] @ (%p)\n", __func__, size, r->m_size, r->m_off, r->m_ptr); 
#endif

    if (add_shared_mem_region(m_area, SHARED_MEM_REGION_SIZE) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: failed to add a new shared mem region\n", __func__);
#endif
      return NULL;
    }

  }

  ptr       = r->m_ptr + r->m_off;
  r->m_off  = r->m_off + size;
  
#ifdef DEBUG
  fprintf(stderr, "%s: allocated [%ld] from sharedmem (%p)\n", __func__, size, ptr);
#endif

  return ptr;
}

/*
  TODO: make this thread safe
*/
void shared_free(void *ptr, size_t size)
{
  struct shared_mem *m;
  struct shared_mem_size *s;
  struct shared_mem_free *f;

  m = m_area;

  if (ptr == NULL || size <= 0)
    return;

#ifdef DEBUG
  fprintf(stderr, "%s: shared free %ld bytes\n", __func__, size);
#endif
  s = find_data_avltree(m->m_free, &size);
  if (s == NULL){

    s = shared_malloc(sizeof(struct shared_mem_size));
    if (s == NULL || size < sizeof(struct shared_mem_free)){
#ifdef DEBUG
      fprintf(stderr, "%s: FAILURE\n", __func__);
#endif
      return;
    }
    
    f = ptr;
    f->f_next = NULL;

    s->s_size = size;
    s->s_top  = f; 
    
    if (store_named_node_avltree(m->m_free, &(s->s_size), s) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: FAILURE\n", __func__);
#endif
      return;
    }

#ifdef DEBUG
    fprintf(stderr, "%s: ptr (%p)\n", __func__, ptr);
#endif

    return;
  }
  
  f = ptr;
  f->f_next = s->s_top;
  s->s_top  = f;

#ifdef DEBUG
  fprintf(stderr, "%s: ptr (%p)\n", __func__, ptr);
#endif

  return;
}


#ifdef TEST_SHARED_MEM
#ifdef DEBUG


#define SIZE      10 
#define CHILD     40

struct test_mem{
  int v;
  char name[100];
};

int main(int argc, char *argv[])
{
  struct test_mem *m[SIZE], *m2[SIZE];
  int i, j;
  pid_t cpid;

  if (create_shared_mem() < 0){
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

  for (i=0; i<SIZE/2; i++){
    shared_free(m[i], sizeof(struct test_mem));
  }
  
  for (i=0; i<SIZE; i++){
    m2[i] = shared_malloc(sizeof(struct test_mem));
    if (m2[i] == NULL){
      fprintf(stderr, "shared_malloc fail\n");
      destroy_shared_mem();
      return 1;
    }
    bzero(m2[i], sizeof(struct test_mem));
  }

  
#if 0
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

#endif
  destroy_shared_mem(); 

  return 0;
}
#endif
#endif
