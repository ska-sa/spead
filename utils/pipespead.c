#define _GNU_SOURCE 
#define _BSD_SOURCE 
#include <stdio.h>
#include <sysexits.h>
#include <errno.h>
#include <unistd.h>
#include <netdb.h>
#include <string.h>
#include <stdint.h>
#include <endian.h>

#include <arpa/inet.h>

#include <sys/types.h>
#include <sys/socket.h>

#include <pcap.h>

#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>



#define BUF 92000

void print_data(unsigned char *buf, int rb)
{
#ifdef DEBUG
  int count, count2;

  count = 0;
  fprintf(stderr, "\t\t   ");
  for (count2=0; count2<30; count2++){
    fprintf(stderr, "%02x ", count2);
  }
  fprintf(stderr,"\n\t\t   ");
  for (count2=0; count2<30; count2++){
    fprintf(stderr, "---");
  }
  fprintf(stderr,"\n\t0x%06x | ", count);
  for (;count<rb; count++){

    fprintf(stderr, "%02X", buf[count]);
    if ((count+1) % 30 == 0){

      if ((count+1) % 900 == 0){
        fprintf(stderr, "\n\n\t\t   ");
        for (count2=0; count2<30; count2++){
          fprintf(stderr, "%02x ", count2);
        }
        fprintf(stderr,"\n\t\t   ");
        for (count2=0; count2<30; count2++){
          fprintf(stderr, "---");
        }
      }

      fprintf(stderr,"\n\t0x%06x | ", count+1);
    } else {
      fprintf(stderr," ");
    }

  }
  fprintf(stderr,"\n");
#endif
}

int setup_udp_streamer(char *host, char *port, struct addrinfo **ai)
{
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

  //freeaddrinfo(res);

  *ai = rp;

  return sfd;
}

void usage(char *argv[])
{
  fprintf(stderr, "%s used to remove any data b4 spead magic and retransmit to UDP\n\tusage %s -[h] dst-host dst-port\n", "|SPEAD", argv[0]);
}

int main(int argc, char *argv[])
{
  int i, j, sfd;
  char c;

  char *host, *port;

  struct addrinfo *ai;


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
  sfd = setup_udp_streamer(host, port, &ai);
  if (sfd < 0)
    return EX_IOERR;

  /*RUN LOOP*/

  //int flag;
  unsigned char *off = NULL;
  //int off;
  //char ned = '\x53';
  uint16_t ipned = htons(0x0800);
  //uint16_t len;
  
#if 0
  struct pcap_pkthdr  *pcappkt;
  struct ether_header *eh;
#endif
  struct iphdr        *iph;
  struct udphdr       *udp;

  int state;

  unsigned char buf[BUF*2];
  int need, have, rb, wb, run, want, pos;

#define S_START   0
#define S_READ    1
#define S_PACKET  2
#define S_DATA    3
  
  state = S_START;

  run = 1;  
  //flag = 0;
  rb = 0;
  need = 0;
  want = BUF;
  pos = 0;
  have = 0;
  off = buf;

  while(run){


    switch (state){

      case S_START:
        rb = read(STDIN_FILENO, buf, sizeof(struct pcap_file_header));
        if (rb <= 0){
          run = 0;
          break;
        }

#ifdef DEBUG
        fprintf(stderr, "pcap hdr s:%ld read: %d bytes\n\tsnap_len: %d\n\tlink-layer-type: %d\n", sizeof(struct pcap_file_header), rb, ((struct pcap_file_header *)buf)->snaplen, ((struct pcap_file_header *)buf)->linktype);
#endif
        state = S_READ;
        break;
      
      case S_READ:
        rb = read(STDIN_FILENO, buf+pos, want);
        if (rb <= 0){
          run = 0;
          break;
        }  
        
#ifdef DEBUG
        fprintf(stderr, "read %d bytes need: %d\n", rb, need);
#endif

        if (need > 0){
          have += rb;
          state = S_DATA;
          break;
        }
        
        state =  S_PACKET;
        break;


      case S_PACKET:
        
        if (rb < sizeof(struct pcap_pkthdr) + sizeof(struct ether_header)+sizeof(struct iphdr)+sizeof(struct udphdr)){
#ifdef DEBUG
          fprintf(stderr, "not enough data to decode\n");
#endif
          state = S_READ;
          break;
        }

        off = memmem(off+sizeof(struct pcap_pkthdr), rb, &ipned, sizeof(ipned));
        if (off){
#ifdef DEBUG
          fprintf(stderr, "eth ip packet id at 0x%lx\n", off-buf);
#endif
          iph = (struct iphdr*) (off+sizeof(uint16_t));

#ifdef DEBUG
          fprintf(stderr, "IP stuff start 0x%lx\n\tver: %d\n\tihl: %d\n\ttos: %d\n\ttotlen :%d\n\tid: %d\n\tfrag_off: %d\n\tttl: %d\n\tproto: %d\n\tcheck: %d\n\tsaddr: 0x%x\n\tdaddr: 0x%x\n", 
            ((void *)iph - (void *)buf), iph->version, iph->ihl, iph->tos, iph->tot_len, iph->id, iph->frag_off, iph->ttl, iph->protocol, iph->check, iph->saddr, iph->daddr);
#endif

          if (iph->protocol == 17){
          
            udp = (struct udphdr*)(off + sizeof(uint16_t) + iph->ihl*sizeof(uint32_t));
#ifdef DEBUG
            fprintf(stderr, "UDP stuff\n\tsport: %d\n\tdport: %d\n\tlen: %d\n\tchksm: 0x%x\n", ntohs(udp->source), ntohs(udp->dest), ntohs(udp->len), ntohs(udp->check));
#endif

            off += sizeof(uint16_t) + iph->ihl*sizeof(uint32_t) + sizeof(struct udphdr);
            need = ntohs(udp->len) - sizeof(struct udphdr);
            have = rb - (off-buf);
#ifdef DEBUG
            fprintf(stderr, "need: %d\nhave: %d off: %ld\n", need, have, off-buf);
#endif
            
            if (need > 0){
              state = S_DATA;
              break;
            }

            state = S_READ;
            
          }
        }
        //print_data(buf, rb);  

        state = S_READ;
        break;
        
      case S_DATA:
         
        if (need <= have){
          
          wb = sendto(sfd, off, need, 0, ai->ai_addr, ai->ai_addrlen);
          if (wb <= 0){
            run =0;
            break;
          }
#ifdef DEBUG
          fprintf(stderr, "sent: %d\n", wb);
#endif
          //print_data(off, need);  

          if (need < have){
            off += wb;
            have -= wb;
            state = S_PACKET;
            need = 0;
#ifdef DEBUG
            fprintf(stderr, "have more try next packet\n\toff: %ld have: %d\n", off-buf, have);
#endif
            break;
          }

          need = 0;
          pos = 0;
          want = BUF;
          off = buf;

        } else if (need > have){
          
          want = need-have;
          pos = rb;

#ifdef DEBUG
          fprintf(stderr, "want %d more pos %d\n", (need-have), pos);
#endif

        }
        
        state = S_READ; 
        break;

    }
  
#if 0
    if (rb > skipsize){
      fprintf(stderr, "ip should start at 0x%x\n", skipsize);
    }

    off = memmem(buf, rb, &ipned, sizeof(ipned));
    if (off){
      fprintf(stderr, "ip packet at 0x%lx\n", off-buf);
    }

    
    count = 0;
    fprintf(stderr,"\t0x%06x | ", count);
    for (;count<rb; count++){

      fprintf(stderr, "%02X", buf[count]);
      if ((count+1) % 20 == 0){
        fprintf(stderr,"\n\t0x%06x | ", count+1);
      } else {
        fprintf(stderr," ");
      }

    }
    fprintf(stderr,"\n");


    off = memmem(buf, rb, &ned, 1);

    if (off){
      
      len = be16toh((uint16_t)(*(off - sizeof(uint16_t)*2)));

      fprintf(stderr, "found needle at 0x%lx try udp len? %d\n", off-buf, len);

      if (rb > (off-buf)){
        wb = sendto(sfd, off, rb-(off-buf), 0, rp->ai_addr, rp->ai_addrlen);
        if (wb <=0){
          run =0;
          break;
        }
      }
    }
#endif

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
