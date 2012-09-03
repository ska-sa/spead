#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>

#include "server.h"
#include "spead_api.h"

#define CALLBACK "spead_api_callback"

void *load_api_user_module(char *mod)
{
  void *mhandle;

  int (*cb)(struct spead_item_group *ig);

  if (mod == NULL)
    return NULL;
  
  mhandle = dlopen(mod, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  if (mhandle == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s\n", dlerror());
#endif
    return NULL;
  }

  cb = dlsym(mhandle, CALLBACK); 
  if (cb == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: module doesn't specify (%s) (%s)\n", __func__, CALLBACK, dlerror());
#endif
    dlclose(mhandle);
    return NULL;
  }
  
#ifdef DEBUG
  fprintf(stderr, "\tAPI module: %s\n", mod);
#endif

  dlclose(mhandle);

  return cb;
}
