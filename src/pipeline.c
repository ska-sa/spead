/* (c) 2012 SKA SA */
/* Released under the GNU GPLv3 - see COPYING */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <unistd.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdint.h>

#include "spead_api.h"
#include "stack.h"

void unload_spead_pipeline(void *data)
{
  struct spead_pipeline *l;
  l = data;
  if (l){
    destroy_stack(l->l_mods, &unload_api_user_module);
    free(l);
  }
}

int load_modules_spead_pipeline(void *so, void *data)
{
  struct stack *s;
  struct spead_api_module *m;
  char *mod;

  if (so == NULL || data == NULL)
    return -1;

  mod = so;
  s = data;

  m = load_api_user_module(mod);
  if (m == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: could not load module <%s> into pipeline\n", __func__, mod);
#endif
    return -1;
  }
  
  if (push_stack(s, m) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: could not add <%s> to stack\n", __func__, mod);
#endif
    unload_api_user_module(m);
    return -1;
  }

  return 0;
}

struct spead_pipeline *create_spead_pipeline(struct stack *pl)
{
  struct spead_pipeline *l;
  
  if (pl == NULL)
    return NULL;
  
  l = malloc(sizeof(struct spead_pipeline));
  if (l == NULL)
    return NULL;
  
  l->l_mods = create_stack();
  if (l->l_mods == NULL){
    destroy_stack(pl, NULL);
    destroy_spead_pipeline(l);
    return NULL;
  }
  
  if (funnel_stack(pl, NULL, &load_modules_spead_pipeline, l->l_mods) < 0){
    destroy_stack(pl, NULL);
    destroy_spead_pipeline(l);
    return NULL;
  }
   
  destroy_stack(pl, NULL);

  return l;
}

void setup_modules_spead_pipeline(void *so, void *data)
{
  struct spead_api_module *m;
  
  if (so == NULL)
    return;

  m = so;

  if (setup_api_user_module(m) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: setup failed for module (%p)\n", __func__, m);
#endif
  }

}

int setup_spead_pipeline(struct spead_pipeline *l)
{
  if (l == NULL)
    return -1;

  traverse_stack(l->l_mods, &setup_modules_spead_pipeline, NULL);
  
  return 0;
}

void destroy_modules_spead_pipeline(void *so, void *data)
{
  struct spead_api_module *m;

  if (so == NULL)
    return;

  m = so;

  if (destroy_api_user_module(m) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: destroy failed for module (%p)\n", __func__, m);
#endif
  }

}

void destroy_spead_pipeline(void *data)
{
  struct spead_pipeline *l;
  l = data;
  if (l){
    traverse_stack(l->l_mods, &destroy_modulues_spead_pipeline, NULL);
  }
}

void run_module_callbacks_spead_pipeline(void *so, void *data)
{
  struct spead_api_module *m;
  struct spead_item_group *ig;

  if (so == NULL)
    return;

  m = so;
  ig = data;

  if (run_api_user_callback_module(m, ig) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: callback failed for module (%p)\n", __func__, m);
#endif
  }

}

int run_callbacks_spead_pipeline(struct spead_pipeline *l, void *data)
{
  if (l == NULL)
    return -1;
  
  traverse_stack(l->l_mods, &run_module_callbacks_spead_pipeline, data);
  
  return 0;
}

void run_module_timers_spead_pipeline(void *so, void *data)
{
  struct spead_api_module *m;

  if (so == NULL)
    return;

  m = so;

  if (run_module_timer_callbacks(m) < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: timer call failed for module (%p)\n", __func__, m);
#endif
  }

}

int run_timers_spead_pipeline(struct spead_pipeline *l)
{
  if (l == NULL)
    return -1;
  
  traverse_stack(l->l_mods, &run_module_timers_spead_pipeline, NULL);
  
  return 0;
}
