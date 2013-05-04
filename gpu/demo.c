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

#include <spead_api.h>

struct demo_o {
  struct data_file        *f;
  struct spead_api_module **mods;
  int                     mcount;
  uint64_t                chunk;
};


void destroy_demo(struct demo_o *d)
{
  int i;

  if (d){
    
    destroy_raw_data_file(d->f);

    if (d->mods){
      for (i=0; i<d->mcount; i++)
        unload_api_user_module(d->mods[i]);
      free(d->mods);
    }

    free(d);
  }
}

struct demo_o *load_demo(int argc, char **argv)
{
  struct demo_o *d;
  char *fname, **mods;
  int modc, i;
  uint64_t chunk;

  if (argc < 4){
#ifdef DEBUG
    fprintf(stderr, "e: usage: %s datafile chunk_size mod1.so [mod2.so] ... [modN.so]\n", argv[0]);
#endif
    return NULL;
  }

  fname = argv[1];
  chunk = atoll(argv[2]);

  mods  = argv+3;
  modc  = argc-3; 

#if 0
  fprintf(stderr, "argv (%p) %s\n", mods, mods[0]); 
  return NULL;
#endif

  d = malloc(sizeof(struct demo_o));
  if (d == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: logic cannot malloc\n");
#endif
    return NULL;
  }

  d->mods = NULL;
  d->chunk = chunk;

  d->f = load_raw_data_file(fname);
  if (d->f == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: cannot load file\n");
#endif
    destroy_demo(d);
    return NULL;
  }

  d->mods = malloc(sizeof(struct spead_api_module *) * modc);
  if (d->mods == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: logic cannot malloc\n");
#endif
    destroy_demo(d);
    return NULL;
  }

  for (i=0; i<modc; i++){
    d->mods[i] = load_api_user_module(mods[i]);
    if (d->mods[i] == NULL){
#ifdef DEBUG
      fprintf(stderr, "e: logic cannot load api_mod\n");
#endif
      d->mcount = i+1;
      destroy_demo(d);
      return NULL;
    }
  }

  d->mcount = modc;

  return d;
}

int setup_pipeline(struct demo_o *d)
{
  int i;

  if (d == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null params\n", __func__);
#endif
    return -1;
  }
  
  for (i=0; i<d->mcount; i++){
    if (setup_api_user_module(d->mods[i]) < 0){
#ifdef DEBUG
      fprintf(stderr, "err mod setup\n");
#endif
    }
  }

  return 0;
}

int run_pipeline(struct demo_o *d, uint64_t chunk)
{
  struct spead_item_group   *ig;
  struct spead_api_item     *itm;

  uint64_t off, have, count, size, rb;
  void *dst;
  int i;

  if (d == NULL || chunk == 0){
#ifdef DEBUG
    fprintf(stderr, "%s: null params\n", __func__);
#endif
    return -1;
  }
  
  off   = 0;
  //chunk = 32*1024;
  //chunk = 64*1024;
  //chunk = 2*1024*1024;
  //chunk = 8;
  //chunk = 1024*1024;
  size  = get_data_file_size(d->f);
  have  = size;

#if 0
  src = get_data_file_ptr_at_off(d->f, off);
  if (src == NULL)
    return -1;
#endif

  ig = create_item_group(chunk, 1);
  if (ig == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: create ig error\n", __func__);
#endif
    return -1;
  }

  itm = new_item_from_group(ig, chunk);
  
  itm->i_valid = 1;
  itm->i_id    = 0xb001;
  itm->i_len   = chunk;
  itm->i_data_len = chunk;
  
  dst = itm->i_data;

  count = 0;
  do {
    
    //memcpy(itm->i_data, src + off, (have < chunk) ? have : chunk);
    if ((rb = request_chunk_datafile(d->f, (have < chunk) ? have : chunk, &dst, NULL)) < 0) {
#ifdef DEBUG
      fprintf(stderr, "%s: error getting chunk\n", __func__);
#endif
      break;
    }
    
    if (size > 0 && copy_to_spead_item(itm, dst, rb) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: error putting chunk\n", __func__);
#endif
      break;
    }

    for (i=0; i<d->mcount; i++){
      if (run_api_user_callback_module(d->mods[i], ig) < 0){
#ifdef DEBUG
        fprintf(stderr, "e: api mod[%d] callback\n", i);
#endif
      }
    }

    off  += chunk;
    have -= chunk;
  
    //fprintf(stderr, "[%ld] at [%ld] of [%ld] left [%ld]\n", count++, off, size, have);

  } while ( (size > 0) ? off < size : 1 );


  destroy_item_group(ig);
    
  return 0;
}

int destroy_pipeline(struct demo_o *d)
{
  int i;

  if (d == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null params\n", __func__);
#endif
    return -1;
  }

  for (i=0; i<d->mcount; i++){
    if (destroy_api_user_module(d->mods[i]) < 0){
#ifdef DEBUG
      fprintf(stderr, "err mod setup\n");
#endif
    }
  }

  return 0;
}


int main(int argc, char *argv[])
{
  struct demo_o *d;

  d = load_demo(argc, argv);
  if (d == NULL)
    return 1;
   
  if (setup_pipeline(d) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: setup pipeline\n");
#endif
    return 1;
  }
  if (run_pipeline(d, d->chunk) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: run pipeline\n");
#endif
    return 1;
  }
  if (destroy_pipeline(d) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: destroy pipeline\n");
#endif
    return 1;
  }

  destroy_demo(d);
    
  return 0;
}
