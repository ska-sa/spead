#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <time.h>
#include <spead_api.h>

#define SPEAD_BF_DATA_ID    0xb000
#define SPEAD_TS_ID 0x1600
#define TS_PER_HEAP 128
 // number of time samples per heap
#define CHANS_PER_HEAP 1024
 // number of channels per heap
#define BYTES_PER_SAMPLE 2
#define FILE_HEAP_LEN 8000

#define EXPECTED_HEAP_LEN TS_PER_HEAP * CHANS_PER_HEAP * BYTES_PER_SAMPLE

typedef unsigned long long ticks;

static __inline__ ticks getticks(void)
{
     unsigned a, d;
     asm("cpuid");
     asm volatile("rdtsc" : "=a" (a), "=d" (d));

     return (((ticks)a) | (((ticks)d) << 32));
}

unsigned long long heap_count;
unsigned long prior_ts;
int w_fd2;

struct snap_shot {
  int flag[4];
  int master_pid;
  unsigned long long prior_ts;
  char filename[100];
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
  unlock_spead_api_module_shared(s);
}


void *spead_api_setup(struct spead_api_module_shared *s)
{
  struct snap_shot *ss;

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
    ss->flag[2] = 1;
    ss->flag[3] = 1;
    ss->prior_ts = 0;
    ss->master_pid = 0;

    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
 
    fprintf(stderr, "%s: PID [%d] decode BF data %s\n", __func__, getpid());

    fflush(stdout);
  
  }
 
  heap_count = 1; 
  prior_ts = 0;

  unlock_spead_api_module_shared(s);

  return NULL;
}



void transpose_bf_block(void *input,void *output, uint64_t data_len, uint64_t stride, uint64_t bytes_per_sample) {
 uint16_t * in;
 uint16_t * out;
 int start_chan = CHANS_PER_HEAP / 4;
 int channels = CHANS_PER_HEAP / 2;
  // only include center half of the band
 int rows = stride;
 int chan,ts,tscols,blocksize,offset_in,offset_out,inner_temp;
 int i,j,k,l;
 uint16_t rr;
 in = input;
 out = output;

#ifdef DEBUG
 fprintf(stderr,"Tranpose: %i cols, %i rows, %i data_len\n",cols,rows,data_len);
#endif
 blocksize = 8;
  // keep this fixed !!
/* for (ts = 0; ts < rows; ts++) {
  for (chan = 0; chan < cols; chan++) {
   out[chan + (ts*cols)] = in[ts + (chan*rows)];
  }
 }*/

 #pragma omp parallel for
 for (ts=0; ts < rows; ts+=blocksize) {
  for (chan=start_chan; chan < (start_chan + channels); chan+=blocksize) {
   offset_in = ts + (chan*rows);
   offset_out = (chan-start_chan) + (ts*channels);
   for (k=0; k < blocksize; k++) {
    inner_temp = k*rows;
    out[offset_out + k] = in[offset_in + 0];
    out[offset_out + k + (1*channels)] = in[offset_in + 1 + inner_temp];
    out[offset_out + k + (2*channels)] = in[offset_in + 2 + inner_temp];
    out[offset_out + k + (3*channels)] = in[offset_in + 3 + inner_temp];
    out[offset_out + k + (4*channels)] = in[offset_in + 4 + inner_temp];
    out[offset_out + k + (5*channels)] = in[offset_in + 5 + inner_temp];
    out[offset_out + k + (6*channels)] = in[offset_in + 6 + inner_temp];
    out[offset_out + k + (7*channels)] = in[offset_in + 7 + inner_temp];
   }
  }
 }
} // end of transpose_bf_block

int write_bf_data(void *data, uint64_t data_len, unsigned long offset)
{
 uint8_t re, im, *dp, *da;
 uint64_t tcks;
 int i;

 dp = malloc(sizeof(uint8_t) * data_len / 2);
  // half the output length as we are only keeping the middle half of the band

 da = data ;

//if (heap_count % 1000) {
// tcks = getticks();
// for (i=0; i < 50000; i++) { 
  transpose_bf_block(da, dp, data_len, TS_PER_HEAP, BYTES_PER_SAMPLE);
// }
// fprintf(stderr,"Ticks Transpose: %llu\n",(getticks() - tcks) / 50000);
 //dp = da;
//}
#ifdef DEBUG
 fprintf(stderr, "Channel 322, timestamp 14: %u %u %u %u",da[28 + (256*322)],da[29 + (256*322)],dp[644 + (14*2048)],dp[645 + (14*2048)]);
#endif

 if (w_fd2 == NULL) { fprintf(stderr,"!!!!! No file !!!!"); return -1;}
 if (pwrite(w_fd2, dp, data_len/2, ((data_len / 2 ) * offset)) < 0) {
  perror("Failed to write.");
  fprintf(stderr,"%s: Failed to write data to disk in thread %d\n",__func__,getpid());
 };

 if (da == NULL || data_len <= 0) {
  fprintf(stderr,"%s: Null data received\n",__func__);
  return -1;
 }
 return 0;
} // end of write_bf_data

int spead_api_callback(struct spead_api_module_shared *s, struct spead_item_group *ig, void *data)
{
  struct snap_shot *ss;
  int file_flag,i;
  unsigned long long ts,ts2;
  struct spead_api_item *itm;
  struct stat sb;

  file_flag = 0;
  itm = NULL;

  itm = get_spead_item_with_id(ig, SPEAD_TS_ID);
  if (itm == NULL){
    fprintf(stderr, "%s: cannot find item with id 0x%x\n", __func__, SPEAD_TS_ID);
    return -1;
  }
   // TS is 5 byte unsigned 
  ts = (int)itm->i_data[0] + (int)itm->i_data[1] * 256 + (int)itm->i_data[2] * 256 * 256 + (int)itm->i_data[3] * 256 * 256 * 256 + (int)itm->i_data[4] * 256 * 256 * 256 * 256;
 
  itm = get_spead_item_with_id(ig, SPEAD_BF_DATA_ID);
  if (itm == NULL){
    fprintf(stderr, "%s: cannot find item with id 0x%x\n", __func__, SPEAD_BF_DATA_ID); 
    return -1;
  }

  // check that the heap size matches our expectations
  if (itm->i_data_len != EXPECTED_HEAP_LEN) {
   fprintf(stderr,"%s: Expecting heap size of %i, got %i\n",__func__,EXPECTED_HEAP_LEN, itm->i_data_len);
   return -1;
  }


  ss = get_data_spead_api_module_shared(s);

  lock_spead_api_module_shared(s);
  if (ss == NULL){
    unlock_spead_api_module_shared(s);
    return -1;
  }
  
  if (ss->flag[2] == 1) {
   fprintf(stderr,"Master thread got flag. Trigger new file...(%d)\n",getpid());
   ss->flag[2] = 0;
   ss->prior_ts = 0;
   ss->flag[1] = 1;
   set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
  }

  if (ss->prior_ts == 0) {
   if (ss->master_pid == 0) ss->master_pid = getpid();
    // make sure we know who is boss
   if (ss->flag[3] == 1) { snprintf(ss->filename,100,"/data1/%llu.dat",ts); ss->flag[3]=1; }
   else { snprintf(ss->filename,100,"/data2/%llu.dat",ts); ss->flag[3]=1;}
   fprintf(stderr,"%s: Requesting new file %s\n",__func__,ss->filename);

   //w_fd2 = open(ss->filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
   //write(w_fd2,"\x93NUMPY\x01\x00V\x00{'descr': '|i1', 'fortran_order': False, 'shape': (128000, 1024, 2), }               \n",96);
   //fprintf(stderr,"Master file open %s\n",ss->filename);
   ss->prior_ts = ts;
   fprintf(stderr,"Reset prior_ts base to %llu. Master pid set to %d\n",ts,ss->master_pid);
   set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
  }

  if (heap_count % FILE_HEAP_LEN == 0 && ss->flag[2] == 0) {
   if (ss->master_pid == getpid()) {
    fprintf(stderr,"Reached file count limit. Setting flag from thread %d (%d)\n",ss->master_pid,getpid());
    ss->flag[2] = 1;
    set_data_spead_api_module_shared(s, ss, sizeof(struct snap_shot));
   }
  }
  unlock_spead_api_module_shared(s);

  if (ss->prior_ts != prior_ts) {
   if (w_fd2 != NULL) close(w_fd2);
   fprintf(stderr, "Opening new file %s with offset base %llu\n",ss->filename, ss->prior_ts);
   if (w_fd2 != NULL) close(w_fd2);
   w_fd2 = open(ss->filename, O_WRONLY | O_CREAT, S_IRWXU | S_IRWXG | S_IRWXO);
   prior_ts = ss->prior_ts;
  }


  /*if (w_fd2 == NULL) { 
   w_fd2 = open(ss->filename, O_WRONLY | O_CREAT, S_IRUSR | S_IWUSR);
   fprintf(stderr,"Opening child file %s with prior_ts %llu\n", ss->filename, ss->prior_ts);
  } */ 

  //fstat(w_fd2, &sb);
  //fprintf(stderr,"\n\nLast file modification %s in thread %d with prior_ts %i\n\n\n", ctime(&sb.st_mtime), getpid(), prior_ts);
  
  heap_count++;

  if (write_bf_data(itm->i_data,itm->i_data_len, (ts - ss->prior_ts) / 4) < 0) {
    fprintf(stderr,"%s: failed to write beamformer data to disk\n",__func__);
    return -1;
  }

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
