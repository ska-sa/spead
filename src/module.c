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
  destroy = dlsym(mhandle, SAPI_DESTROY); 
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
  m->m_setup   = setup;

  m->m_data    = NULL;

#if 0
  m->m_data    = (*setup)();
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
  char *err;

  if (m){
#if 0   
    if (m->m_destroy){
      (*m->m_destroy)(m->m_data);
    }
#endif

    if (m->m_handle){
      dlclose(m->m_handle);
      if((err = dlerror()) != NULL){
#ifdef DEBUG
        fprintf(stderr, "%s: dlerror <%s>\n", __func__, err);
#endif
      }

    }

    free(m);
     
  }

#ifdef DEBUG
  fprintf(stderr, "%s: done\n", __func__);
#endif
}

int setup_api_user_module(struct spead_api_module *m)
{
  if (m){
    if (m->m_setup){
      m->m_data = (*m->m_setup)();
#ifdef DEBUG
      fprintf(stderr, "%s: module (%p) data @ (%p)\n", __func__, m, m->m_data);
#endif
      return 0;
    }
  }
  return -1;
}

int destroy_api_user_module(struct spead_api_module *m)
{
  if (m){
    if (m->m_destroy){
      (*m->m_destroy)(m->m_data);
#ifdef DEBUG
      fprintf(stderr, "%s: module (%p) data destroy called\n", __func__, m);
#endif
      return 0;
    }
  }

  return -1;
}

int run_api_user_callback_module(struct spead_api_module *m, struct spead_item_group *ig)
{
  if (m){
    if (m->m_cdfn){
      return (*m->m_cdfn)(ig, m->m_data);
    }
  }
  return -1; 
}
