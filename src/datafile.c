/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <sys/stat.h>
#include <sys/mman.h>

#include "spead_api.h"



struct data_file *load_raw_data_file(char *fname)
{
  struct data_file *f;

  f = malloc(sizeof(struct data_file));
  if (f == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }
  
  f->f_m    = 0;
  f->f_off  = 0;
  f->f_fd   = 0;
  f->f_fmap = NULL;

  if (stat(fname, &(f->f_fs)) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: stat error: %s\n", __func__, strerror(errno));
#endif
    destroy_raw_data_file(f);
    return NULL;
  }
  
  f->f_fd = open(fname, O_RDONLY);
  if (f->f_fd < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: open error: %s\n", __func__, strerror(errno));
#endif
    destroy_raw_data_file(f);
    return NULL;
  }

  f->f_fmap = mmap(NULL, f->f_fs.st_size, PROT_READ, MAP_PRIVATE, f->f_fd, 0);
  if (f->f_fmap == MAP_FAILED){
#ifdef DEBUG
    fprintf(stderr, "%s: mmap error: %s\n", __func__, strerror(errno));
#endif
    destroy_raw_data_file(f);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "%s: mapped <%s> into (%p) size [%ld] bytes\n", __func__, fname, f->f_fmap, (uint64_t) (f->f_fs.st_size));
#endif

  return f;
}

void destroy_raw_data_file(struct data_file *f)
{
  if (f){
    if (f->f_fmap)
      munmap(f->f_fmap, f->f_fs.st_size);

    if (f->f_fd)
      close(f->f_fd);

    free(f);
  }
}

size_t get_data_file_size(struct data_file *f)
{
  if (f){
    return f->f_fs.st_size;
  }
  return 0;
}

void *get_data_file_ptr_at_off(struct data_file *f, uint64_t off)
{
  if (f == NULL)
    return NULL;

  if (f->f_fs.st_size <= off){
    return NULL;
  }

  return f->f_fmap + off;
}

int request_chunk_datafile(struct data_file *f, uint64_t len, void **ptr)
{
  int rtn;

  if (f == NULL || len < 0 || ptr == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: error params\n", __func__);
#endif
    return -1; 
  }
  
  lock_mutex(&(f->f_m));

  if ((*ptr = get_data_file_ptr_at_off(f, f->f_off)) == NULL){
    unlock_mutex(&(f->f_m));
#ifdef DEBUG
    fprintf(stderr, "%s: EOF\n", __func__);
#endif
    return 0;
  }
 
  rtn       = (f->f_fs.st_size > f->f_off+len) ? len : f->f_fs.st_size - f->f_off;
  f->f_off += rtn;

  unlock_mutex(&(f->f_m));

  return rtn;
}

