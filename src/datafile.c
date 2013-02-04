/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <libgen.h>

#include <sys/stat.h>
#include <sys/mman.h>

#include "spead_api.h"


struct data_file *create_raw_data_file(char *fname)
{
  struct data_file *f;

  f = malloc(sizeof(struct data_file));
  if (f == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic cannot malloc\n", __func__);
#endif
    return NULL;
  }
  
  f->f_name = fname;
  f->f_m    = 0;
  f->f_off  = 0;
  f->f_fd   = 0;
  f->f_fmap = NULL;
  f->f_state= DF_FILE;

  return f;
}

struct data_file *write_raw_data_file(char *fname)
{
  struct data_file *f;
  long flags;
  
  flags = 0;

  f = create_raw_data_file(fname);
  if (f == NULL)
    return NULL;

  if (strncmp(fname, "-", 1) == 0){
    f->f_fd = STDOUT_FILENO;

    flags = fcntl(f->f_fd, F_GETFD, 0);
    
    flags &= ~O_NONBLOCK;

    if (fcntl(f->f_fd, F_SETFD, flags) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error unsetting fcntl o_nonblock\n", __func__);
#endif
      destroy_raw_data_file(f);
      return NULL;
    }

    f->f_state = DF_STREAM;

  } else {
    f->f_fd = open(fname, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
    if (f->f_fd < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: open error (%s)\n", __func__, strerror(errno));
#endif
      destroy_raw_data_file(f);
      return NULL;
    }
  }

  return f;
}

struct data_file *load_raw_data_file(char *fname)
{
  struct data_file *f;

  f = create_raw_data_file(fname);
  if (f == NULL)
    return NULL;

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

    if (f->f_fd && f->f_state != DF_STREAM)
      close(f->f_fd);

    free(f);
  }
}

int write_chunk_raw_data_file(struct data_file *f, uint64_t off, void *src, uint64_t len)
{
  int64_t sw, pos, wb;

  if (f == NULL || src == NULL || len <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }

  sw = len;
  pos = 0;
  wb = 0;

  switch(f->f_state){
    case DF_FILE:
      
      do { 
        wb = pwrite(f->f_fd, src + pos, sw, off + pos);
        if (wb <= 0){
          if (wb < 0){
            switch(errno){
              case EINTR:
              case EAGAIN:
                continue;
              case ESPIPE:
              default:
#ifdef DEBUG
                fprintf(stderr, "%s: pwrite error (%s)\n", __func__, strerror(errno));
#endif
                return -1;
            }
          } else {
            /*should never reach here*/
            if (pos == len)
              break;
          }
        } else {
          pos += wb;
          sw  -= wb;
        }
      } while(wb == len || sw <= 0);
      
      break;

    case DF_STREAM:
      
      wb = write(f->f_fd, src, len);
      if (wb < 0) {
#ifdef DEBUG
        fprintf(stderr, "%s: stream write error (%s)\n", __func__, strerror(errno));
#endif
        return -1;
      }

      break;
  
  }

#ifdef DEBUG
  fprintf(stderr, "%s: worte [%ld] bytes to <%s> @ [%ld]\n", __func__, len, f->f_name, off);
#endif

  return 0;
}

int write_next_chunk_raw_data_file(struct data_file *f, void *src, uint64_t len)
{
  if (f == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: param error\n", __func__);
#endif
    return -1;
  }
  
  lock_mutex(&(f->f_m));
  
  if (write_chunk_raw_data_file(f, f->f_off, src, len) < 0){
    unlock_mutex(&(f->f_m));
    return -1;
  }

  f->f_off += len;
  
  unlock_mutex(&(f->f_m));
  
  return 0;
}

size_t get_data_file_size(struct data_file *f)
{
  if (f){
    return f->f_fs.st_size;
  }
  return 0;
}

char *get_data_file_name(struct data_file *f)
{
  return (f) ? basename(f->f_name) : NULL;
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

uint64_t request_chunk_datafile(struct data_file *f, uint64_t len, void **ptr, uint64_t *chunk_off_rtn)
{
  uint64_t rtn;

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

  if (chunk_off_rtn)
    *chunk_off_rtn = f->f_off;
 
  rtn       = (f->f_fs.st_size > f->f_off+len) ? len : f->f_fs.st_size - f->f_off;
  f->f_off += rtn;

  unlock_mutex(&(f->f_m));

  return rtn;
}

