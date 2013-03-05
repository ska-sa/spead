#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <math.h>

#include <spead_api.h>

#include "shared.h"

#define KERNELS_FILE  "/kernels.cl"

#define SPEAD_BF_DATA_ID    0xb000

struct snap_shot {
  int flag[2];
};

struct sapi_object {
  struct ocl_ds *o_ds;
  struct ocl_kernel *o_power;
};

void destroy_sapi_object(void *data)
{
  struct sapi_object *so;
  so = data;
  if (so == NULL){
    destroy_ocl_kernel(so->o_power);
    destroy_ocl_ds(so->o_ds);
    shared_free(so, sizeof(struct sapi_object));
  }
}

void spead_api_destroy(struct spead_api_module_shared *s, void *data)
{
  struct snap_shot *ss;
  
  lock_spead_api_module_shared(s);

  if ((ss = get_data_spead_api_module_shared(s)) != NULL){ 

    shared_free(ss, sizeof(struct snap_shot));

    clear_data_spead_api_module_shared(s);
#ifdef DEBUG
    fprintf(stderr, "%s: PID [%d] destroyed spead_api_shared\n", __func__, getpid());
#endif

  } else {

#ifdef DEBUG
    fprintf(stderr, "%s: PID [%d] spead_api_shared is clean\n", __func__, getpid());
#endif

  }

  unlock_spead_api_module_shared(s);

  destroy_sapi_object(data); 
}


void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct snap_shot *ss;
  struct sapi_object *so;

  so = NULL;
  ss = NULL;

  lock_spead_api_module_shared(s);

  if (!(ss = get_data_spead_api_module_shared(s))){ 

    ss = shared_malloc(sizeof(struct snap_shot));
    if (ss == NULL){
      unlock_spead_api_module_shared(s);
      return NULL;
    }

    ss->flag[0] = 0;
    ss->flag[1] = 0;

    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
  
#ifdef DEBUG
    fprintf(stderr, "%s: PID [%d] created spead_api_shared\n", __func__, getpid());
#endif

    fprintf(stdout, "set term x11 size 1280,720\nset view map\n");
    fflush(stdout);
  
  }

  unlock_spead_api_module_shared(s);

  so = shared_malloc(sizeof(struct sapi_object));
  if (so == NULL)
    return NULL;
  
  so->o_ds = create_ocl_ds(KERNELDIR KERNELS_FILE);
  if (so->o_ds == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  so->o_power = create_ocl_kernel(so->o_ds, "power");
  if (so->o_power == NULL){
    destroy_sapi_object(so);
    return NULL;
  }

  return so;
}

int format_bf_data_hack(void *data, uint64_t data_len)
{
#define TIMESAMPLES       128
#define BYTES_PER_SAMPLE  2
  int time, chan;

  uint64_t chans = data_len / BYTES_PER_SAMPLE / TIMESAMPLES;
  uint8_t re, im, *da;

#ifdef DEBUG
  fprintf(stderr, "%s: chans %ld\n", __func__, chans);
#endif
  
  da = data;

  if (da == NULL || data_len <= 0)
    return -1;

#if 0
  fprintf(stdout, "splot '-' matrix with image\n");
  for (time=0; time < TIMESAMPLES; time++){
    for(chan=0; chan < chans; chan++){
      
      re = da[time+chan*TIMESAMPLES];
      im = da[time+chan*TIMESAMPLES+1];

      fprintf(stdout, "%f ", hypotf((float)re, (float)im));

    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "e\ne\n");
  fflush(stdout);
#endif

  

  return 0; 
}

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct snap_shot *ss;
  struct sapi_object *so;
  int flag;
  struct spead_api_item *itm;

  flag = 0;
  itm = NULL;

  so = data;
  if (so == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: opencl not available\n", __func__);
#endif
    return -1;
  }

  ss = get_data_spead_api_module_shared(s);

  lock_spead_api_module_shared(s);
  if (ss == NULL){
    unlock_spead_api_module_shared(s);
    return -1;
  }

  if (ss->flag[0]){
    flag = 1;
    ss->flag[0] = 0;
  }
  
  if (flag && ig != NULL){
    
#ifdef DEBUG
    fprintf(stderr, "%s: PID %d got the flag\nitems=[%ld] size=[%ld]\n", __func__, getpid(), ig->g_items, ig->g_size);
#endif

    while ((itm = get_next_spead_item(ig, itm))){
      
      if (itm->i_id == SPEAD_BF_DATA_ID){
#ifdef DEBUG
        fprintf(stderr, "%s: id [\033[32m%s\033[0m\t0x%x] data size [%ld]\n", __func__, hr_spead_id(itm->i_id), itm->i_id, itm->i_data_len);
#endif

        if (format_bf_data_hack(itm->i_data, itm->i_data_len) < 0){
#ifdef DEBUG
          fprintf(stderr, "%s: error formatting data\n", __func__);
#endif
        }
        break;
        
      }
    }
  }

  unlock_spead_api_module_shared(s);
  return 0;
}

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{
  struct snap_shot *ss;
  
  lock_spead_api_module_shared(s);

  ss = get_data_spead_api_module_shared(s);
  if (ss == NULL){
    unlock_spead_api_module_shared(s);
    return -1; 
  }

  ss->flag[0] = 1;

  unlock_spead_api_module_shared(s);

  return 0;
}
