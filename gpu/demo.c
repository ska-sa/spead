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
  struct stat             fs;
  int                     fd;
  uint8_t                 *fmap;
  struct spead_api_module **mods;
  int                     mcount;
};


void destroy_demo(struct demo_o *d)
{
  int i;

  if (d){
    if (d->fd)
      close(d->fd);
    if (d->fmap)
      munmap(d->fmap, d->fs.st_size);
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

  if (argc < 3){
#ifdef DEBUG
    fprintf(stderr, "e: usage: %s datafile mod1.so [mod2.so] ... [modN.so]\n", argv[0]);
#endif
    return NULL;
  }

  fname = argv[1];

  mods  = argv+2;
  modc  = argc-2; 

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

  if (stat(fname, &(d->fs)) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: stat error: %s\n", strerror(errno));
#endif
    return NULL;
  }
  
  d->fd = open(fname, O_RDONLY);
  if (d->fd < 0){
#ifdef DEBUG
    fprintf(stderr, "e: open error: %s\n", strerror(errno));
#endif
    return NULL;
  }

  d->fmap = mmap(NULL, d->fs.st_size, PROT_READ, MAP_PRIVATE, d->fd, 0);
  if (d->fmap == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: mmap error: %s\n", strerror(errno));
#endif
    close(d->fd);
    return NULL;
  }

#ifdef DEBUG
  fprintf(stderr, "mapped <%s> into (%p) size [%d] bytes\n", fname, d->fmap, (int) (d->fs.st_size));
#endif
  
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

int run_pipeline(struct demo_o *d)
{
  int i;

  if (d == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: null params\n", __func__);
#endif
    return -1;
  }
  
  for (i=0; i<d->mcount; i++){
    


  }
  
  
  



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

#if 0
  struct sapi_o *a;
#endif

  uint64_t off, chunk, have;
  
  d = load_demo(argc, argv);
  if (d == NULL)
    return 1;
  
   
  if (setup_pipeline(d) < 0){
#ifdef DEBUG
    fprintf(stderr, "e: setup pipeline\n");
#endif
    return 1;
  }
  if (run_pipeline(d) < 0){
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


#if 0
  a = spead_api_setup();
  if (a == NULL){
#ifdef DEBUG
    fprintf(stderr, "e: spead api setup\n"); 
#endif
    munmap(data, fs.st_size);
    close(fd);
    return 1;
  }

  off   = 0;
  chunk = 1024;
  have  = fs.st_size;

  do {

    print_data(data+off, (have < chunk) ? have : chunk);

    off += chunk;
    have -= chunk;

#ifdef DEBUG
    fprintf(stderr, "\n");
#endif

  } while (off < fs.st_size);

  spead_api_destroy(a);
#endif


  destroy_demo(d);
    
  return 0;
}
