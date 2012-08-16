#define _GNU_SOURCE 
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>

#include <sys/types.h>
#include <sys/socket.h>

#if 0
#include <pcap/pcap.h>

#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#endif


#define BUF 10000

void usage(char *argv[])
{
  fprintf(stderr, "%s used to remove any data b4 spead magic and retransmit to UDP\n\tusage %s -[h] dst-host dst-port\n", "|SPEAD", argv[0]);
}

int main(int argc, char *argv[])
{
  int i, j, rb, wb, run;
  char c;

  char *host, *port;

  unsigned char buf[BUF];

  /*PARAMETER STREAMER*/
  i=1;
  j=1;
  host = NULL;
  port = NULL;

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
          usage(argv);
          return EX_OK;

        /*settings*/
#if 0
        case 'p':
        case 'w':
        case 'b':
        case 'l':
          j++;
          if (argv[i][j] == '\0'){
            j = 0;
            i++;
          }
          if (i >= argc){
            fprintf(stderr, "%s: option -%c requires a parameter\n", argv[0], c);
            return EX_USAGE;
          }
          switch (c){
            case 'p':
              port = argv[i] + j;  
              break;
            case 'w':
              cpus = atoll(argv[i] + j);
              break;
            case 'b':
              hashes = atoll(argv[i] + j);
              break;
            case 'l':
              fprintf(stderr, "param l\n");
              break;
          }
          i++;
          j = 1;
          break;
#endif

        default:
          fprintf(stderr, "%s: unknown option -%c\n", argv[0], c);
          return EX_USAGE;
      }

    } else {
     
      if (host == NULL){
        host = argv[i];
      } else if (port == NULL) {
        port = argv[i];
      } else {
        fprintf(stderr, "%s: extra argument %s\n", argv[0], argv[i]);
        return EX_USAGE;
      }

      i++;
    }
    
  }

  if (!host || !port){
    usage(argv);
    return EX_USAGE;
  }

  /*UDP STREAMER SETUP*/
  struct addrinfo hints;
  struct addrinfo *res, *rp;

  int sfd;

  uint64_t reuse_addr;
  
  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family     = AF_UNSPEC;
  hints.ai_socktype   = SOCK_DGRAM;
  hints.ai_flags      = AI_PASSIVE;
  hints.ai_protocol   = 0;
  hints.ai_canonname  = NULL;
  hints.ai_addr       = NULL;
  hints.ai_next       = NULL;
  
  if ((reuse_addr = getaddrinfo(host, port, &hints, &res)) != 0) {
#ifdef DEBUG
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(reuse_addr));
#endif
    return -1;
  }

  for (rp = res; rp != NULL; rp = rp->ai_next) {
#if DEBUG>1
    fprintf(stderr, "%s: res (%p) with: %d\n", __func__, rp, rp->ai_protocol);
#endif
    if (rp->ai_family == AF_INET6)
      break;
  }

  rp = (rp == NULL) ? res : rp;

  sfd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
  if (sfd < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error socket\n", __func__);
#endif
    freeaddrinfo(res);
    return -1;
  }

  /*set reuse addr*/
  reuse_addr   = 1;
  setsockopt(sfd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr));

  /*make the send buffer really big!*/
  reuse_addr = 1024*1024*1024;
  if (setsockopt(sfd, SOL_SOCKET, SO_SNDBUF, &reuse_addr, sizeof(reuse_addr)) < 0){
#ifdef DEBUG
    fprintf(stderr,"%s: error setsockopt: %s\n", __func__, strerror(errno));
#endif
  }

  freeaddrinfo(res);

  /*RUN LOOP*/

  int flag;
  int skipsize;
  unsigned char *off;
  int count;
  char ned = '\x53';

  //skipsize = sizeof(struct ether_header) + sizeof(struct iphdr) + sizeof(struct udphdr);
  run = 1;  
  flag = 0;

  while(run){
  
    rb = read(STDIN_FILENO, buf, BUF);
    if (rb <= 0){
      run = 0;
      break;
    }
    fprintf(stderr, "read %d bytes\n", rb);

    off = memmem(buf, rb, &ned, 1);

    if (off){
      fprintf(stderr, "found needle at %d\n", off-buf);

      if (rb > (off-buf)){
        wb = sendto(sfd, off, rb-(off-buf), 0, rp->ai_addr, rp->ai_addrlen);
        if (wb <=0){
          run =0;
          break;
        }
      }
    }

#if 0
    switch(flag){
      case 0:
        off   = skipsize + sizeof(struct pcap_file_header);      
        if (rb == sizeof(struct pcap_file_header)){
          fprintf(stderr, "throw away pcap file header\n");
          flag  = 1;
          continue;
        }

      case 1:
        off = skipsize;
        break;
    }
    
    if (rb > off){
      wb = sendto(sfd, buf+off, rb-off, 0, rp->ai_addr, rp->ai_addrlen);
      if (wb <=0){
        run =0;
        break;
      }
      count = 0;
      fprintf(stderr, "wrote %d bytes\n\t0x%06x | ", wb, count);
      for (;count<wb; count++){

        fprintf(stderr, "%02X", buf[off+count]);
        if ((count+1) % 20 == 0){
          fprintf(stderr,"\n\t0x%06x | ", count);
        } else {
          fprintf(stderr," ");
        }

      }
      fprintf(stderr,"\n");

    }
#endif
  }

  close(sfd);

  fprintf(stderr, "Exiting cleanly\n");

  return 0;
}
