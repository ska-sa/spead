#include <stdio.h>
#include <signal.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include <spead_api.h>

#define S_RATE     1024 
#define BUF_SIZE   (S_RATE * 2)
#define FREQ_HZ    300

static int run = 1;

void handle_us(int signum) 
{
  run = 0;
}

int register_signals_us()
{
  struct sigaction sa;

  sigfillset(&sa.sa_mask);
  sa.sa_handler   = handle_us;
  sa.sa_flags     = 0;

  if (sigaction(SIGINT, &sa, NULL) < 0)
    return -1;

  if (sigaction(SIGTERM, &sa, NULL) < 0)
    return -1;

  return 0;
}

int main(int argc, char *argv[])
{
  struct data_file *df;
  unsigned char buf[BUF_SIZE];
  int i;

  if (register_signals_us() < 0){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error\n", __func__);
#endif
    return -1;
  }

  /**write output to stdout*/
  df = write_raw_data_file("-");
  if (df == NULL){
#ifdef DEBUG
    fprintf(stderr, "%s: logic error\n", __func__);
#endif
    return -1;
  }

  float phase    = 0.0; 
  float freq_rad = FREQ_HZ * 2 * M_PI / S_RATE;

  fprintf(stderr, "%s: radians: %f\n", __func__, freq_rad);

  for (i=0; i<BUF_SIZE; i++){
    phase += freq_rad;
    buf[i] = (unsigned char)((sin(phase)+1)*128);
    //fprintf(stdout,"%d ", buf[i]);
  }

  while (run){

    if (write_next_chunk_raw_data_file(df, buf, BUF_SIZE) < 0){
#ifdef DEBUG
      fprintf(stderr, "%s: failed to write\n", __func__);
#endif
      run = 0;
      break;
    }
  }

  destroy_raw_data_file(df);
#ifdef DEBUG
  fprintf(stderr, "%s:DONE\n", __func__);
#endif

  return 0;
}
