#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>

#include <spead_api.h>

#define F_ID_SPEAD_ID 4096
#define TIMESTAMP_SPEAD_ID 4097
#define DATA_SPEAD_ID 4098
#define DEFAULT_PREFIX "/data1/latest"
#define DEFAULT_FILENAME "spead_out"
#define FILE_HEAP_LEN 8000 

typedef enum { false, true } bool;

int w_fd2;
unsigned long long heap_cnt;
int file_count;

struct snap_shot {
  unsigned long long prior_ts;
  char filename[255];
  char * prefix;
  unsigned long long offset;
  unsigned long long data_len;
  int id;
  bool with_markers;

  int file_count;
  int master_id;
  int new_file;
};

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
  close(w_fd2);
  printf("EXITING");
  unlock_spead_api_module_shared(s);
}

void *spead_api_setup(struct spead_api_module_shared *s)
{

  char * env_temp;

  struct snap_shot *ss;  
  // State of stream??
  ss = NULL;

  lock_spead_api_module_shared(s); 
  //lock access to shared resources??

  if (!(ss = get_data_spead_api_module_shared(s))){ 
  // If stream state hasn't been initialised, this must be the master thread...

    ss = shared_malloc(sizeof(struct snap_shot)); 
    //allocate memory for shared state
    if (ss == NULL){  // if mem allocation failed??
      unlock_spead_api_module_shared(s); //unloack spead resources??
      return NULL;
    }

    env_temp = getenv("PREFIX");
    //Set data file prefix from environment variables, or just use default of /Data1/latest
    if (env_temp != NULL) ss->prefix = env_temp;
    else ss->prefix = DEFAULT_PREFIX;

    env_temp = getenv("FILENAME");
    //Set data file prefix from environment variables, or just use default of /Data1/latest
    if (env_temp != NULL) snprintf(ss->filename,255, env_temp);
    else snprintf(ss->filename,255, DEFAULT_FILENAME);

    env_temp = getenv("SAVEID");
    if (env_temp != NULL) ss->id = atoi(env_temp);
    else ss->id = -1;

    env_temp = getenv("WITHMARKERS");
    //If WITHMARKERS is set, then we print markers in the data file. Each packet will have DEAD00000000DEAD printed at the end
    //Each item will have 0000DEADDEAD0000 printed after it 
    if (env_temp != NULL) ss->with_markers = true;
    else ss->with_markers = false;

    fprintf(stderr, "CONFIGURATION\n---------------------------------\nPREFIX=%s\nFILENAME=%s\nSAVEID=%d\nWITHMARKERS?%s\n",
      ss->prefix, ss->filename, ss->id, ss->with_markers==true?"TRUE":"FALSE");

    ss->master_id = 0;
    ss->new_file = 4;
    ss->file_count = 0;
    

    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));

  }

  unlock_spead_api_module_shared(s);
  //Unlock shared resources

  heap_cnt = 0;
  file_count = 0;

  return NULL;
}


int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct snap_shot *ss;
  struct spead_api_item *itm;
  unsigned long long local_offset;
  bool release = false;

  ss = get_data_spead_api_module_shared(s);
  //Get shared data
  
  itm = NULL;

  if(ss->master_id == getpid())
    fprintf (stderr, "HEAP COUNT = %llu\n", heap_cnt);

  fprintf (stderr," --------------------Saving data to %s---------------------------\n", ss->filename);

  lock_spead_api_module_shared(s);
  //lock shared resources
  if (ss == NULL){
    fprintf(stderr,"ss is null");
    fflush(stdout);
    unlock_spead_api_module_shared(s);
    //Shared resources empty, unlock and return
    return -1;
  }

  if(ss->master_id == 0){
    fprintf(stderr,"MASTER THREAD %d with file_count %d\n", getpid(), file_count);
    ss->master_id = getpid();
  }

  if (w_fd2 == NULL || ss->file_count != file_count){ 
    if (ss->master_id == getpid() && w_fd2 == NULL){
      file_count++;
      fprintf(stderr,"MASTER THREAD %d set file_count to %d\n", getpid(), ss->file_count);
    }
    file_count = ss->file_count;
    char new_file_name[255];
    if (w_fd2 != NULL) close(w_fd2);
    sprintf(new_file_name, "%s/%s%04d.dat",ss->prefix,ss->filename, ss->file_count);
    w_fd2 = open(new_file_name, O_WRONLY | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
  }

  if (ss->data_len != NULL){
    local_offset = ss->offset;
    ss->offset = ss->offset + ss->data_len;
    heap_cnt++;

    if (heap_cnt % FILE_HEAP_LEN == 0 && ss->master_id == getpid())
    {
      ss->file_count++;
      fprintf(stderr,"MASTER THREAD %d set file_count to %d\n", getpid(), ss->file_count);
      ss->offset = 0;
    }
    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
    unlock_spead_api_module_shared(s);
    //Already know the length of an itemgroup, can safely unlock resources
  }
  else{
    //Need to calculate the length of an itemgroup
    local_offset=0;
    release = true;
    heap_cnt++;
  }


  if (ss->id == -1){
    while ((itm = get_next_spead_item(ig, itm))){
      //while there is still data in the itemgroup
      
#ifdef DEBUG
        print_spead_item(itm);
        print_data(itm->i_data, itm->i_data_len);
#endif

      if (write_data(itm->i_data, itm->i_data_len, local_offset)) {
        fprintf(stderr,"%s: failed to write item data to disk\n",__func__);
        return -1;
      }

      local_offset = local_offset + itm->i_data_len;

      if (ss->with_markers == true && w_fd2 != NULL){
        if (write_data("\x00\x00\xde\xad\xde\xad\x00\x00", 8, local_offset)) {
          fprintf(stderr,"%s: failed to write marker data to disk\n",__func__);
          return -1;
        }

        local_offset = local_offset + 8;
      }
    }
  }
  else{

    itm = get_spead_item_with_id(ig, ss->id);

#ifdef DEBUG
    print_spead_item(itm);
    print_data(itm->i_data, itm->i_data_len);
#endif

    if (write_data(itm->i_data, itm->i_data_len, local_offset)) {
      fprintf(stderr,"%s: failed to write item data to disk\n",__func__);
      return -1;
    }

    local_offset = local_offset + itm->i_data_len;
  }

  if (ss->with_markers == true && w_fd2 != NULL){
    if (write_data("\xde\xad\x00\x00\00\00\xde\xad", 8, local_offset)) {
      fprintf(stderr,"%s: failed to write marker data to disk\n",__func__);
      return -1;
    }
    local_offset = local_offset + 8;
  }

  if (release){
    ss->data_len = local_offset;
    ss->offset = local_offset;
    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
    unlock_spead_api_module_shared(s);
  }

  return 0;
}

int write_data(void *data, uint64_t data_len, unsigned long offset)
{
 uint8_t *dp;

 fprintf(stderr, "WRITING TO at %llu + %llu = %llu for len\n",offset, data_len, offset + data_len);

 dp = data;

 if (dp == NULL || data_len <= 0) {
  fprintf(stderr,"%s: Null data received\n",__func__);
  return -1;
 }

 unsigned long written = 0;

 if (w_fd2 == NULL) { fprintf(stderr,"!!!!! No file !!!!"); return -1;}
 if (written = pwrite(w_fd2, dp, data_len, offset) < 0) {
  perror("Failed to write.");
  fprintf(stderr,"%s: Failed to write data to disk in thread %d\n",__func__,getpid());
 }

 return 0;
} // end of write_bf_data

int spead_api_timer_callback(struct spead_api_module_shared *s, void *data)
{
  // struct snap_shot *ss;
  
  // lock_spead_api_module_shared(s);

  // ss = get_data_spead_api_module_shared(s);
  // if (ss == NULL){
  //   unlock_spead_api_module_shared(s);
  //   return -1; 
  // }

  // unlock_spead_api_module_shared(s);

  return 0;
}
