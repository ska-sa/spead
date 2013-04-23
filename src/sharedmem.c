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
#include "mutex.h"

static struct shared_mem *m_area = NULL;

#if 0
int compare_shared_mem_regions(const void *v1, const void *v2)
{
  if (*(uint64_t *)v1 < *(uint64_t *)v2)
    return -1;
  else if (*(uint64_t *)v1 > *(uint64_t *)v2)
    return 1;
  return 0;
}
#endif

int compare_shared_mem_size(const void *v1, const void *v2)
{
  if (*(size_t *) v1 < *(size_t *)v2)
    return -1;
  else if (*(size_t *) v1 > *(size_t *)v2)
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
    fprintf(stderr, "%s: mmap error 0 %s\n", __func__, strerror(errno));
#endif
    return NULL;
  }
  
  r = mmap(NULL, sizeof(struct shared_mem_region), PROT_WRITE | PROT_READ, MAP_SHARED | MAP_ANONYMOUS, (-1), 0);
  if (r == MAP_FAILED){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error 1 %s\n", __func__, strerror(errno));
#endif
    munmap(ptr, size);
    return NULL;
  }

  if (ptr == NULL || r == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error 2 %s\n", __func__, strerror(errno));
#endif
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: r(%p) ptr(%p)\n", __func__,  r, ptr);
#endif
  
  r->r_m    = 0;
  r->r_id   = mid;
  r->r_size = size;
  r->r_off  = 0;
  r->r_next = NULL;
  r->r_ptr  = ptr;

#ifdef DEBUG
  fprintf(stderr, "%s: created %ld byte shared mem region from (%p - %p)\n", __func__, size, r->r_ptr, r->r_ptr + size); 
#endif
  
  return r;
}

void destroy_shared_mem_region(void *data)
{ 
  struct shared_mem_region *r = data;
  if (r){
    if (r->r_ptr)
      munmap(r->r_ptr, r->r_size);
    munmap(r, sizeof(struct shared_mem_region));
  }
}

int add_shared_mem_region(struct shared_mem *m, uint64_t size)
{
  struct shared_mem_region *r;

  if (m == NULL || size <= 0)
    return -1;

  lock_mutex(&(m->m_m));

  r = create_shared_mem_region(m->m_next_id, size);
  if (r == NULL){
    unlock_mutex(&(m->m_m));
    return -1;
  }

#if 0
  if (store_named_node_avltree(m->m_tree, &(r->r_id), r) < 0){
    destroy_shared_mem_region(r);
    unlock_mutex(&(m->m_m));
    return -1;
  }
#endif
 
  lock_mutex(&(r->r_m));
  r->r_next = m->m_current;
  unlock_mutex(&(r->r_m));

  m->m_current = r;
  m->m_next_id++;

  unlock_mutex(&(m->m_m));

  return 0;
}

int create_shared_mem()
{
  struct shared_mem *m;
  struct avl_tree *f;

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

#if 0
  t = create_avltree(&compare_shared_mem_regions);
  if (t == NULL){
    destroy_shared_mem_region(m->m_current);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }
#endif

  f = create_avltree(&compare_shared_mem_size);
  if (f == NULL){
    destroy_shared_mem_region(m->m_current);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }

#if 0
  if (store_named_node_avltree(t, &(m->m_current->r_id), m->m_current) < 0){
    destroy_shared_mem_region(m->m_current);
    destroy_avltree(t, NULL);
    munmap(m, sizeof(struct shared_mem));
    return -1;
  }
#endif

  m->m_m       = 0;
  m->m_free    = f;
  m->m_next_id = 1;

  return 0;
}

void destroy_shared_mem()
{
  struct shared_mem_region *r;
  int i;
  if (m_area) {
#ifdef DEBUG
    fprintf(stderr, "%s: mem regions %ld free nodes %ld\n", __func__, m_area->m_next_id, m_area->m_free->t_ncount);
#endif

    for (i=0; (r = m_area->m_current) != NULL; i++){
      if (r){

        m_area->m_current = r->r_next;
#ifdef DEBUG
        fprintf(stderr, "%s: freeing (%p) with [%ld] bytes for mem region [%ld]\n", __func__, r, r->r_size, r->r_id);
#endif
        destroy_shared_mem_region(r);

      }    
    }

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

  if (m_area == NULL)
    create_shared_mem();

  if (size < 0 || size < sizeof(struct shared_mem_free)){
#ifdef DEBUG
    fprintf(stderr, "%s: \033[31ma shared memory segment must have a positive size and > sizeof(struct shared_mem_free) %ld\033[0m\n", __func__, sizeof(struct shared_mem_free)); 
#endif
    return NULL;
  }

  s = find_data_avltree(m_area->m_free, &size);
  if (s != NULL) {
#if DEBUG>1
    fprintf(stderr, "%s: found a free group list (%p) for size %ld s_top (%p)\n", __func__, s, size, s->s_top);
#endif
    
    lock_mutex(&(s->s_m));
    
    if (s->s_top){
      
      f = s->s_top;
      ptr = s->s_top;

      s->s_top = f->f_next;

      unlock_mutex(&(s->s_m));

#if DEBUG>1
      fprintf(stderr, "%s: ptr (%p)\n", __func__, ptr);
#endif
      
      return ptr;
    }

    unlock_mutex(&(s->s_m));

#if DEBUG>1
    fprintf(stderr, "%s: no nodes in list\n", __func__);
#endif

  }
 
  r = get_current_shared_mem_region(m_area);
  if (r == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: shared memory region is NULL\n",__func__);
#endif
    return NULL;
  }

  lock_mutex(&(r->r_m));

  if ((size + r->r_off) > r->r_size){
#ifdef DEBUG
    fprintf(stderr, "%s: WARN shared_malloc size req [%ld] mem stats msize [%ld] m_off [%ld] @ (%p)\n", __func__, size, r->r_size, r->r_off, r->r_ptr); 
#endif

    if (add_shared_mem_region(m_area, SHARED_MEM_REGION_SIZE) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: failed to add a new shared mem region\n", __func__);
#endif
      unlock_mutex(&(r->r_m));
      return NULL;
    }

    unlock_mutex(&(r->r_m));

    r = get_current_shared_mem_region(m_area);
    if (r == NULL){
#ifdef DEBUG
      fprintf(stderr, "%s: shared memory region is NULL\n",__func__);
#endif
      return NULL;
    }

    lock_mutex(&(r->r_m));

  }

  ptr       = r->r_ptr + r->r_off;
  r->r_off  = r->r_off + size;
  
  unlock_mutex(&(r->r_m));

#if DEBUG>1
  fprintf(stderr, "%s: allocated [%ld] from sharedmem (%p)\n", __func__, size, ptr);
#endif

  return ptr;
}

void shared_free(void *ptr, size_t size)
{
  struct shared_mem *m;
  struct shared_mem_size *s;
  struct shared_mem_free *f;

  m = m_area;

  if (ptr == NULL || size <= 0)
    return;

#if DEBUG>1
  fprintf(stderr, "%s: shared free %ld bytes\n", __func__, size);
#endif
  s = find_data_avltree(m->m_free, &size);
  if (s == NULL){

    s = shared_malloc(sizeof(struct shared_mem_size));
    if (s == NULL || size < sizeof(struct shared_mem_free)){
#ifdef DEBUG
      fprintf(stderr, "%s: FAILURE null or size %ld < shared_mem_free_size %ld\n", __func__, size, sizeof(struct shared_mem_free));
#endif
      return;
    }

    s->s_m    = 0;

    lock_mutex(&(s->s_m));

#if DEBUG>1
    fprintf(stderr, "%s: created free size group s (%p)\n", __func__, s); 
#endif

    f = (struct shared_mem_free *) (ptr);
    f->f_next = NULL;

    s->s_size = size;
    s->s_top  = f; 

    if (store_named_node_avltree(m->m_free, &(s->s_size), s) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: FAILURE cannot store size group in free tree\n", __func__);
#endif  
      shared_free(s, sizeof(struct shared_mem_size));
      unlock_mutex(&(s->s_m));
      return;
    }

    unlock_mutex(&(s->s_m));
#if DEBUG>1
    fprintf(stderr, "%s: 1 ptr (%p)\n", __func__, ptr);
#endif

    return;
  }
  
#if DEBUG>1
  fprintf(stderr, "%s: found free size group s (%p)\n", __func__, s); 
#endif

  lock_mutex(&(s->s_m));
  f = (struct shared_mem_free *) (ptr);
  f->f_next = s->s_top;
  s->s_top  = f;
  unlock_mutex(&(s->s_m));
  
#if DEBUG>1
  fprintf(stderr, "%s: 2 ptr (%p) s_top (%p) fnext (%p)\n", __func__, ptr, s->s_top, f->f_next);
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

#ifdef DEBUG
  fprintf(stderr, "%s: info sizeof(shared_mem_free) %ld\n", __func__, sizeof(struct shared_mem_free));
#endif

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
    m[i] = NULL;
  }
  
  if (add_shared_mem_region(m_area,SHARED_MEM_REGION_SIZE) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: err adding new shared mem region \n", __func__);
#endif
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

  for (i=0; i<SIZE; i++){
    shared_free(m[i], sizeof(struct test_mem));
    shared_free(m2[i], sizeof(struct test_mem));
    m[i] = NULL;
    m2[i] = NULL;
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
