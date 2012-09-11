#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>

#include "server.h"
#include "spead_api.h"

struct spead_api_module *load_api_user_module(char *mod)
{
  void *mhandle;
  struct spead_api_module *m;

  int (*cb)(struct spead_item_group *ig);
  void *(*setup)();
  int (*destroy)(void *data);

  if (mod == NULL){
#ifdef DEBUG
    fprintf(stderr, "null module name\n");
#endif
    return NULL;
  }

  mhandle = dlopen(mod, RTLD_NOW | RTLD_LOCAL);
  if (mhandle == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s\n", dlerror());
#endif
    return NULL;
  }

  cb = dlsym(mhandle, SAPI_CALLBACK); 
  if (cb == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_CALLBACK, dlerror());
#endif
    dlclose(mhandle);
    return NULL;
  }
  setup = dlsym(mhandle, SAPI_SETUP); 
  if (setup == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_SETUP, dlerror());
#endif
    dlclose(mhandle);
    return NULL;
  }
  destroy = dlsym(mhandle, SAPI_CALLBACK); 
  if (destroy == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_DESTROY, dlerror());
#endif
    dlclose(mhandle);
    return NULL;
  }
  
  m = malloc(sizeof(struct spead_api_module));
  if (m == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not malloc\n", __func__);
#endif
    dlclose(mhandle);
    return NULL;
  }

  m->m_handle  = mhandle;
  m->m_cdfn    = cb;
  m->m_destroy = destroy;

  m->m_data    = (*setup)();
#if 0
  if (m->m_data == NULL){
    unload_api_user_module(m);
    return NULL;
  }
#endif

#ifdef DEBUG
  fprintf(stderr, "\tDATA sink:\t%s\n", mod);
#endif

  return m;
}

void unload_api_user_module(struct spead_api_module *m)
{
  if (m){
    
    if (m->m_destroy){
      
      (*m->m_destroy)(m->m_data);

    }

    if (m->m_handle){
      
      dlclose(m->m_handle);

    }

    free(m);
     
  }
}
