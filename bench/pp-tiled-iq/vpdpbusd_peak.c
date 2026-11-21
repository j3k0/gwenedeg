// The core's vpdpbusd zmm peak (PP-RESULTS.md): 16 independent accumulators. gcc -O2 -march=native vpdpbusd_peak.c
#include <immintrin.h>
#include <stdio.h>
#include <time.h>
int main(){
 struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
 long n=200000000;
 __asm__ volatile(
 "vpxord %%zmm0,%%zmm0,%%zmm0\n vpternlogd $0xff,%%zmm30,%%zmm30,%%zmm30\n vmovdqa64 %%zmm30,%%zmm31\n"
 "mov %0,%%rcx\n1:\n"
 "vpdpbusd %%zmm30,%%zmm31,%%zmm0\n vpdpbusd %%zmm30,%%zmm31,%%zmm1\n vpdpbusd %%zmm30,%%zmm31,%%zmm2\n vpdpbusd %%zmm30,%%zmm31,%%zmm3\n"
 "vpdpbusd %%zmm30,%%zmm31,%%zmm4\n vpdpbusd %%zmm30,%%zmm31,%%zmm5\n vpdpbusd %%zmm30,%%zmm31,%%zmm6\n vpdpbusd %%zmm30,%%zmm31,%%zmm7\n"
 "vpdpbusd %%zmm30,%%zmm31,%%zmm8\n vpdpbusd %%zmm30,%%zmm31,%%zmm9\n vpdpbusd %%zmm30,%%zmm31,%%zmm10\n vpdpbusd %%zmm30,%%zmm31,%%zmm11\n"
 "vpdpbusd %%zmm30,%%zmm31,%%zmm12\n vpdpbusd %%zmm30,%%zmm31,%%zmm13\n vpdpbusd %%zmm30,%%zmm31,%%zmm14\n vpdpbusd %%zmm30,%%zmm31,%%zmm15\n"
 "dec %%rcx\n jnz 1b\n" :: "r"(n) : "rcx","xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","xmm30","xmm31");
 clock_gettime(CLOCK_MONOTONIC,&t1);
 double t=(t1.tv_sec-t0.tv_sec)+1e-9*(t1.tv_nsec-t0.tv_nsec);
 printf("%.3f ns per dpbusd, %.1f GMAC/s per core\n", t*1e9/(n*16.0), n*16.0*64/t/1e9); }
