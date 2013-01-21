/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <string.h>
#include <unistd.h>
#include <endian.h>
#include <sys/types.h>
#include <sys/mman.h>

#include "hash.h"
#include "spead_api.h"
#include "server.h"

#if 0
struct spead_api_item *init_spead_api_item(struct spead_api_item *itm, int vaild, int id, int len, unsigned char *data)
{
  if (itm == NULL)
    return NULL;

  itm->i_valid = vaild;
  itm->i_id    = id;
  itm->i_len   = len;
  itm->i_data  = data;

  return itm;
}
#endif

struct spead_item_group *create_item_group(uint64_t datasize, uint64_t nitems)
{
  struct spead_item_group *ig;
  
  //if (datasize <= 0 || nitems <= 0)
  if (nitems <= 0)
    return NULL;

  ig = malloc(sizeof(struct spead_item_group));
  if (ig == NULL)
    return NULL;

  ig->g_off   = 0;
  ig->g_items = 0;

#if 0
  ig->g_size  = datasize + nitems*(sizeof(struct spead_api_item));
#endif

  /*TODO: note the size*/  
  //ig->g_size  = datasize + nitems*(sizeof(struct spead_api_item) + sizeof(uint64_t));
  ig->g_size  = datasize + nitems*(sizeof(struct spead_api_item));

#if 0
  ig->g_map   = shared_malloc(ig->g_size);
  if (ig->g_map == NULL){
    free(ig);
#ifdef DEBUG
    fprintf(stderr, "%s: failed to get shared_malloc for ig map\n", __func__);
#endif
    return NULL;
  }
#endif

#if 1
  ig->g_map   = mmap(NULL, ig->g_size, PROT_WRITE | PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, (-1), 0);
  if (ig->g_map == MAP_FAILED){
    free(ig);
    return NULL;
  }
#endif

  bzero(ig->g_map, ig->g_size);

#ifdef DEBUG
  fprintf(stderr, "%s: CREATE ITEM GROUP [%ld] items map size [%ld] bytes\n", __func__, nitems, ig->g_size);
#endif

  return ig;
}

struct spead_api_item *new_item_from_group(struct spead_item_group *ig, uint64_t size)
{
  struct spead_api_item *itm;
  uint64_t item_size;

  item_size = size + sizeof(struct spead_api_item);
  
  if (ig == NULL || size <= 0){
#ifdef DEBUG
    fprintf(stderr, "%s: param error size: %ld\n", __func__ ,size);
#endif
    return NULL;
  }
    
  if ((ig->g_off + item_size) > ig->g_size){
#ifdef DEBUG
    fprintf(stderr, "%s: parameter error (ig->g_off + size) %ld ig->g_size %ld\n", __func__, ig->g_off+item_size, ig->g_size);
#endif
    return NULL;
  }

  itm = (struct spead_api_item *) (ig->g_map + ig->g_off);
  if (itm == NULL)
    return NULL;

  ig->g_off += item_size;
  ig->g_items++;

#if DEBUG>2
  fprintf(stderr, "GROUP map (%p): size %ld offset: %ld data: %p\n", ig->g_map, ig->g_size, ig->g_off, itm->i_data);
#endif

  itm->i_valid    = 0;
  itm->i_id       = 0;
  itm->io_data    = NULL;
  itm->io_size    = 0;
  itm->i_len      = size;
  itm->i_data_len = 0;

  return itm;
}

#if 0 
int grow_spead_item_group(struct spead_item_group *ig, uint64_t datasize, uint64_t nitems)
{
  uint64_t oldsize;
  void *oldmap;

  if (ig == NULL){
#ifdef DEUBG
    fprintf(stderr, "%s: parameter error\n", __func__);
#endif
    return -1;
  }

  oldsize = ig->g_size;
  oldmap  = ig->g_map;

  ig->g_items += nitems;
  ig->g_size  += datasize + nitems*(sizeof(struct spead_api_item) + sizeof(uint64_t));
  
  ig->g_map   = mremap(ig->g_map, oldsize, ig->g_size, MREMAP_MAYMOVE);
  if (ig->g_map == MAP_FAILED){
    ig->g_map   = oldmap;
    ig->g_size  = oldsize;
#ifdef DEBUG
    fprintf(stderr, "%s: mremap failed\n", __func__);
#endif
    return -1;
  }

  return 0;
}
#endif

void destroy_item_group(struct spead_item_group *ig)
{
  if (ig){
    
    /*TODO: this is we must do something with shared_mem*/
#if 1
    if (ig->g_map)
      munmap(ig->g_map, ig->g_size); 
#endif

    free(ig);
  }
}

struct spead_api_item *get_spead_item_at_off(struct spead_item_group *ig, uint64_t off)
{
  struct spead_api_item *itm;

  if (ig == NULL){
#if DEBUG>1
    fprintf(stderr, "%s: null params\n", __func__);
#endif  
    return NULL;
  }

  if (off >= ig->g_size){
#if DEBUG>1
    fprintf(stderr, "%s: off [%ld] >= ig size [%ld]\n", __func__, off, ig->g_size);
#endif  
    return NULL;
  }

  itm = (struct spead_api_item *) (ig->g_map + off);

  if (itm->i_len == 0){
#if DEBUG>1
    fprintf(stderr, "%s: item has 0 lenght\n", __func__);
#endif  
    return NULL;
  }

  return itm;
}

struct spead_api_item *get_next_spead_item(struct spead_item_group *ig, struct spead_api_item *current)
{
  static uint64_t off = 0;

  if (current == NULL){
    off = 0;
#if DEBUG> 1
    fprintf(stderr, "%s: pid [%d] off now: %ld\n", __func__, getpid(), off);
#endif
    return get_spead_item_at_off(ig, 0);
  }

  off += sizeof(struct spead_api_item) + current->i_len;
#if DEBUG> 1
  fprintf(stderr, "%s: pid [%d] off now: %ld\n", __func__, getpid(), off);
#endif
  return get_spead_item_at_off(ig, off);
}

int set_spead_item_io_data(struct spead_api_item *itm, void *ptr, size_t size)
{
  if (itm){

    if (ptr == itm->io_data){
#ifdef DEBUG
      fprintf(stderr, "%s: io_data pointer match\n", __func__);
#endif
      return 0;
    }

    if (itm->io_data){
#if DEBUG> 2
      fprintf(stderr, "%s: WARN overwriting io_data ptr\n", __func__);
#endif
    }
    
    itm->io_data = ptr;
    itm->io_size = size;

    return 0;
  }

  return -1;
}

int copy_to_spead_item(struct spead_api_item *itm, void *src, size_t len)
{
  if (itm == NULL || src == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: cannot operate with null params\n", __func__);
#endif
    return -1;
  }
  
  memcpy(itm->i_data, src, len);

  //itm->i_len = len;
  itm->i_data_len = len;

  return len;
}

int set_item_data_ones(struct spead_api_item *itm)
{
  if (itm){
    itm->i_id = SPEAD_ONES_ID;
    itm->i_data_len = itm->i_len;
    memset(itm->i_data, 0x11, itm->i_len);    
    return 0;
  }
  return -1; 
}

int set_item_data_zeros(struct spead_api_item *itm)
{
  if (itm){
    itm->i_id = SPEAD_ZEROS_ID;
    itm->i_data_len = itm->i_len;
    memset(itm->i_data, 0, itm->i_len);    
    return 0;
  }
  return -1; 
}

int set_item_data_ramp(struct spead_api_item *itm)
{
  uint64_t count;
  register int n;
  unsigned char *buf;

  if (itm){
    
    itm->i_id = SPEAD_RAMP_ID;

    buf = itm->i_data;

    count = itm->i_len;
    itm->i_data_len = itm->i_len;
    n = (count+15)/16;
      
    switch (count % 16){
      case 0: do { 
               *(buf++) = 0;
      case 15: *(buf++) = 1;
      case 14: *(buf++) = 2;
      case 13: *(buf++) = 3;
      case 12: *(buf++) = 4;
      case 11: *(buf++) = 5;
      case 10: *(buf++) = 6;
      case 9:  *(buf++) = 7;
      case 8:  *(buf++) = 8;
      case 7:  *(buf++) = 9;
      case 6:  *(buf++) = 10;
      case 5:  *(buf++) = 11;
      case 4:  *(buf++) = 12;
      case 3:  *(buf++) = 13;
      case 2:  *(buf++) = 14;
      case 1:  *(buf++) = 15;
              } while (--n > 0);
    }

    return 0;
  }
  return -1; 
}



