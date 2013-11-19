#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <fcntl.h>
#include <sysexits.h>
#include <signal.h>
#include <netdb.h>
#include <math.h>

#include <sys/mman.h>
    
#include <sys/types.h>
#include <sys/socket.h>

    
#include "spead_api.h"

static volatile int run = 1;

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

int usage(char **argv)
{
  fprintf(stderr, "usage:\n\t%s (options) destination port"
                  "\n\n\tOptions"
                  "\n\t\t-h help"
                  "\n\t\t-s sender"
                  "\n\t\t-r receiver\n\n", argv[0]);
  return EX_USAGE;
}


int run_sender(struct spead_socket *x)
{
  int rb;
  


  while(run){
    


  }

  
  return 0;
}

int run_receiver(struct spead_socket *x)
{
  
  
  
  return 0;
}



int main(int argc, char *argv[])
{
  int i=1,j=1,k=0;
  char c, flag = 0, *host = NULL, *port = NULL;

  struct spead_socket *x = NULL;

  if (argc < 2)
    return usage(argv);

  while (i < argc){
    if (argv[i][0] == '-'){

      c = argv[i][j];

      switch(c){
        case '\0':
          j = 1;
          i++;
          break;
        case '-':
          j++;
          break;

        /*switches*/  
        case 'h':
          return usage(argv);

        case 'r':
          j++;
          flag = 1;
          k = 1;
          break;

        case 's':
          j++;
          flag = 0;
          break;

          /*settings*/
        
        default:
          fprintf(stderr, "%s: unknown option -%c\n", argv[0], c);
          return EX_USAGE;
      }

    } else {
      /*parameters*/
      switch (k){
        case 0:
          host = argv[i];
          k++;
          break;
        case 1:
          port = argv[i];
          k++;
          break;
        default:
          fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
          return EX_USAGE;
      }
      i++;
      j=1;
    }
  }

  if (k < 2){
    fprintf(stderr, "%s: insufficient arguments\n", __func__);
    return EX_USAGE;
  }

  if (register_signals_us() < 0)
    return EX_SOFTWARE;

  x = create_spead_socket(host, port);
  if (x == NULL){
    return EX_SOFTWARE;
  }

  switch (flag){
    
    case 0: /*sender*/
#ifdef DEBUG
      fprintf(stderr, "%s: running sender\n", __func__);
#endif

#if 0
      if (connect_spead_socket(x) < 0){
        goto cleanup;
      }
#endif 

#if 1
      if (run_sender(x) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: run sender fail\n", __func__);
#endif
      }
#endif

      break;

    case 1: /*receiver*/
#ifdef DEBUG
      fprintf(stderr, "%s: running receiver\n", __func__);
#endif

#if 0
      if (bind_spead_socket(x) < 0){
        goto cleanup;
      }
#endif

#if 1
      if (run_receiver(x) < 0){
#ifdef DEBUG
        fprintf(stderr, "%s: run receiver fail\n", __func__);
#endif
      }
#endif

      break;
  }

cleanup:  
  destroy_spead_socket(x);
  
  destroy_shared_mem();
#ifdef DEBUG
  fprintf(stderr,"%s: done\n", __func__);
#endif
  return 0;
}

