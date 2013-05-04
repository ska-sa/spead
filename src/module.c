/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

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
  struct spead_api_module_shared *s;

  int (*cb)(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data);
  void *(*setup)(struct spead_api_module_shared *s);
  int (*destroy)(struct spead_api_module_shared *s, void *data);
  int (*timer)(struct spead_api_module_shared *s, void *data);

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
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_CALLBACK, dlerror());
    dlclose(mhandle);
    return NULL;
  }
  setup = dlsym(mhandle, SAPI_SETUP); 
  if (setup == NULL){
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_SETUP, dlerror());
    dlclose(mhandle);
    return NULL;
  }
  destroy = dlsym(mhandle, SAPI_DESTROY); 
  if (destroy == NULL){
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_DESTROY, dlerror());
    dlclose(mhandle);
    return NULL;
  }
  timer = dlsym(mhandle, SAPI_TIMER_CALLBACK); 
  if (timer == NULL){
    fprintf(stderr, "%s: module doesn't implement (%s) (%s)\n", __func__, SAPI_TIMER_CALLBACK, dlerror());
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
  m->m_timer   = timer;

  m->m_data    = NULL;

  m->m_s       = shared_malloc(sizeof(struct spead_api_module_shared));
  if (m->m_s == NULL){
    unload_api_user_module(m);
    return NULL;
  }

  s = m->m_s;

  s->s_m          = 0;
  s->s_data       = NULL;
  s->s_data_size  = 0;

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

#if 0
void unload_api_user_module(struct spead_api_module *m)
#endif
void unload_api_user_module(void *data)
{
  struct spead_api_module *m;
  char *err;
  
  m = data;

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
    if (m->m_s){
      shared_free(m->m_s, sizeof(struct spead_api_module_shared));
      m->m_s = NULL;
    } else {
#ifdef DEBUG
      fprintf(stderr, "%s: spead api shared module is already null\n", __func__);
#endif
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
      m->m_data = (*m->m_setup)(m->m_s);
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
      (*m->m_destroy)(m->m_s, m->m_data);
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
#if DEBUG>1
      fprintf(stderr, "%s: about to call api callback\n", __func__);
#endif
      return (*m->m_cdfn)(m->m_s, ig, m->m_data);
    }
#ifdef DEBUG
    fprintf(stderr, "%s: null call back function\n", __func__);
#endif
  }
  return -1; 
}


void lock_spead_api_module_shared(struct spead_api_module_shared *s)
{
  if (s){
    lock_mutex(&(s->s_m));
  }
}

void unlock_spead_api_module_shared(struct spead_api_module_shared *s)
{
  if (s){
    unlock_mutex(&(s->s_m));
  }
}

void *get_data_spead_api_module_shared(struct spead_api_module_shared *s)
{
  return (s)?s->s_data:NULL;
}

void set_data_spead_api_module_shared(struct spead_api_module_shared *s, void *data, size_t size)
{
  if (s){
    s->s_data = data;
    s->s_data_size = size;
  }
}

size_t get_data_size_spead_api_module_shared(struct spead_api_module_shared *s)
{
  return (s)?s->s_data_size:0;
}

void clear_data_spead_api_module_shared(struct spead_api_module_shared *s)
{
  if (s){
    s->s_data = NULL;
    s->s_data_size = 0;
  }
}

int run_module_timer_callbacks(struct spead_api_module *m)
{
  if (m){
    if (m->m_timer){
      return (*m->m_timer)(m->m_s, m->m_data);
    }
  }
  return -1;
}
