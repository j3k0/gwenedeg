#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
int main(int argc,char**argv){
  size_t mb = argc>1? atol(argv[1]):1024; size_t n = mb<<20;
  uint8_t *a = aligned_alloc(64,n);
  #pragma omp parallel for
  for(size_t i=0;i<n;i+=4096) memset(a+i,1,4096);
  int reps=10; double best=1e9; long long sink=0;
  for(int r=0;r<reps;r++){
    double t0=now(); __m512i acc=_mm512_setzero_si512();
    #pragma omp parallel
    { __m512i s=_mm512_setzero_si512();
      #pragma omp for nowait
      for(size_t i=0;i<n;i+=256){
        s=_mm512_add_epi64(s,_mm512_load_si512(a+i));
        s=_mm512_add_epi64(s,_mm512_load_si512(a+i+64));
        s=_mm512_add_epi64(s,_mm512_load_si512(a+i+128));
        s=_mm512_add_epi64(s,_mm512_load_si512(a+i+192));
      }
      #pragma omp critical
      acc=_mm512_add_epi64(acc,s);
    }
    double dt=now()-t0; if(dt<best)best=dt; sink+=_mm512_reduce_add_epi64(acc);
  }
  printf("threads=%d size=%zuMB read BW=%.1f GB/s (sink %lld)\n",omp_get_max_threads(),mb,n/best/1e9,sink&1);
  return 0;
}
