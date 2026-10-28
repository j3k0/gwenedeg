/* vdpbf16ps against its definition (the bf16 dot-product attention, src/kernels_avx512.h; TODO G4 (b)): `make
 * bf16dp-test` builds build/bf16dp_test (this CPU's instruction, if it has AVX512_BF16) and build/bf16dp_test-emu (the
 * kernel's emulation for AVX-512 CPUs without it: two vfmadd per lane); make test runs both (group kv).
 *   build/bf16dp_test [LANES]   (default 2^24 random lanes per distribution)
 * Per dword lane the definition is two FMAs, the odd bf16 element first: c = fma(a1, b1, c); c = fma(a0, b0, c), each
 * rounded to nearest even, DAZ on every input (a denormal bf16 factor or fp32 accumulator reads as a signed zero), FTZ
 * on each result (tiny after rounding -> zero of the result's sign: x86's MXCSR FTZ / DAZ rules). Checked:
 *   1. edge cases with their expected results (the order of the two products, the two roundings, ties, DAZ of factors
 *      and accumulator, FTZ with tininess after rounding (2^-126 - 2^-151 is not flushed, 2^-126 - 2^-150 is), signed
 *      zeros, overflow, NaN propagation and quieting, inf * 0);
 *   2. random lanes in 6 distributions (all bit patterns; products cancelling an accumulator ~1; around 2^-126;
 *      near overflow; mixed exponents; denormal accumulators and factors): the kernel's att512_dp (the instruction, or
 *      the emulation in the -emu build), the C twin of the generic build (two fmaf under MXCSR FTZ / DAZ, as
 *      forward.c's attention_vec_dot) and an exact reference independent of MXCSR (double FMA + explicit DAZ / FTZ),
 *      all bit-identical (NaN results: NaN in each; the payloads of the twin and the instruction agree here too);
 *   3. the conversions: att512_cvt_bf16 (vcvtneps2bf16, or its integer emulation) == f32_bf16_daz (round to nearest
 *      even after flushing a denormal to a zero of its sign) on all 2^32 fp32 inputs, and att512_f16_pairs ==
 *      f32_bf16(f16 -> fp32) on all 2^16 f16 values in both halves.
 * Exit status 0 if all equal. Measured on the dev box (AMD EPYC Genoa, Zen 4): all equal, 2^24 lanes x 6 in ~3 s. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../src/kernels_avx512.h"

#if !defined(RUNQ_ATT512)
int main(void) {
    printf("SKIP bf16dp_test: not an AVX-512 build\n");
    return 0;
}
#else

static float fb(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float b2f(uint16_t h) { return fb((uint32_t)h << 16); }

// the instruction itself (any build: the target attribute), if the CPU has it
__attribute__((target("avx512f,avx512bf16"), noinline)) static void hw16(float *r, const float *c, const uint32_t *a, const uint32_t *b) {
    __m512 x = _mm512_dpbf16_ps(_mm512_loadu_ps(c), (__m512bh)_mm512_loadu_si512(a), (__m512bh)_mm512_loadu_si512(b));
    _mm512_storeu_ps(r, x);
}
// the kernel's lane function (att512_dp: the instruction, or two vfmadd in the -emu build)
__attribute__((noinline)) static void kern16(float *r, const float *c, const uint32_t *a, const uint32_t *b) {
    _mm512_storeu_ps(r, att512_dp(_mm512_loadu_ps(c), _mm512_loadu_si512(a), _mm512_loadu_si512(b)));
}
// the generic build's C twin: two fmaf (forward.c, att_dot2), under MXCSR FTZ / DAZ
__attribute__((noinline)) static float twin(float c, uint32_t a, uint32_t b) {
    c = fmaf(b2f((uint16_t)(a >> 16)), b2f((uint16_t)(b >> 16)), c);
    return fmaf(b2f((uint16_t)a), b2f((uint16_t)b), c);
}
// the exact reference, independent of MXCSR: double FMA (the bf16 product is exact, one rounding to 53 bits: rounding
// that to 24 bits equals rounding the exact value, also in the tininess decision), explicit DAZ / FTZ
static float daz(float x) { uint32_t u = bits(x); return (u & 0x7f800000u) ? x : fb(u & 0x80000000u); }
static float step_x(float c, float a, float b) {
    a = daz(a); b = daz(b); c = daz(c);
    double x = fma((double)a, (double)b, (double)c);
    if (fabs(x) < 0x1p-126 - 0x1p-151) // tiny after rounding to 24 bits (the tie 2^-126 - 2^-151 rounds up to 2^-126)
        return signbit(x) ? -0.0f : 0.0f;
    return (float)x;
}
static float ref(float c, uint32_t a, uint32_t b) {
    c = step_x(c, b2f((uint16_t)(a >> 16)), b2f((uint16_t)(b >> 16)));
    return step_x(c, b2f((uint16_t)a), b2f((uint16_t)b));
}

static int fails = 0;
static void check(int ok, const char *what, const char *detail) {
    printf("%s %s  %s\n", ok ? "PASS" : "FAIL", what, detail);
    fails += !ok;
}

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static inline uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static uint16_t rbf(int emin, int emax) { // random bf16, exponent field in [emin, emax]
    int e = emin + (int)(rnd() % (uint64_t)(emax - emin + 1));
    return (uint16_t)((rnd() & 1) << 15 | (unsigned)e << 7 | (rnd() & 0x7f));
}
static uint16_t b16(float f) { return (uint16_t)(bits(f) >> 16); } // (f exactly a bf16 value)
#define PAIR(lo, hi) ((uint32_t)(uint16_t)(lo) | (uint32_t)(uint16_t)(hi) << 16)

int main(int argc, char **argv) {
    long n = argc > 1 ? atol(argv[1]) : 1L << 24;
    int has_hw = __builtin_cpu_supports("avx512bf16");
    unsigned csr0 = _mm_getcsr();
    _mm_setcsr(csr0 | 0x8040); // FTZ | DAZ, as forward.c's att_ftz_on for the attention
    printf("# bf16dp_test: att512_dp = %s; the CPU %s AVX512_BF16\n", ATT512_DP_HW ? "vdpbf16ps" : "the FMA emulation",
           has_hw ? "has" : "lacks (the instruction is not checked)");
    char d[512];

    // 1. edge cases: (c, a, b, expected)
    uint16_t one = b16(1.0f), z = 0;
    struct { float c; uint32_t a, b, want; const char *what; } E[] = {
        { 1.0f, PAIR(b16(0x1p-12f), b16(0x1p-12f)), PAIR(b16(0x1p-12f), b16(0x1p-12f)), 0x3f800000u, "two roundings: 1 + 2^-24 + 2^-24 -> 1 (one rounding: 1 + 2^-23)" },
        { 1.0f, PAIR(b16(0x1.8p-24f), b16(0x1p-24f)), PAIR(one, one), 0x3f800001u, "odd element first: 1 + 2^-24 (tie, even) + 1.5 * 2^-24 -> 1 + 2^-23" },
        { 1.0f, PAIR(b16(0x1p-24f), b16(0x1.8p-24f)), PAIR(one, one), 0x3f800002u, "odd element first: 1 + 1.5 * 2^-24 + 2^-24 (tie, even) -> 1 + 2^-22" },
        { 0.0f, PAIR(z, 0x0001), PAIR(z, b16(0x1p100f)), 0x00000000u, "DAZ: a bf16 denormal factor (odd) reads as 0" },
        { 0.0f, PAIR(0x8001, z), PAIR(b16(0x1p100f), z), 0x00000000u, "DAZ: a bf16 denormal factor (even), -0 * x + 0 -> +0" },
        { fb(0x00080000u), PAIR(z, b16(0x1p-63f)), PAIR(z, b16(0x1p-63f)), 0x00800000u, "DAZ: a denormal accumulator 2^-130 reads as 0" },
        { fb(0x80080000u), PAIR(0x8000, 0x8000), PAIR(one, one), 0x80000000u, "DAZ keeps the sign: -denormal -> -0, + -0 + -0 -> -0" },
        { 0.0f, PAIR(z, b16(0x1p-65f)), PAIR(z, b16(0x1p-65f)), 0x00000000u, "FTZ: 2^-130 -> 0" },
        { 0.0f, PAIR(b16(0x1p-65f), z), PAIR(b16(-0x1p-65f), z), 0x80000000u, "FTZ keeps the sign: -2^-130 -> -0" },
        { 0x1p-126f, PAIR(z, b16(0x1p-75f)), PAIR(z, b16(-0x1p-76f)), 0x00800000u, "FTZ after rounding: 2^-126 - 2^-151 rounds to 2^-126, kept" },
        { 0x1p-126f, PAIR(z, b16(0x1p-75f)), PAIR(z, b16(-0x1p-75f)), 0x00000000u, "FTZ after rounding: 2^-126 - 2^-150 (24 bits exactly) is tiny -> 0" },
        { 0x1p-126f, PAIR(z, b16(0x1.8p-76f)), PAIR(z, b16(-0x1p-76f)), 0x00800000u, "FTZ after rounding: 2^-126 - 1.5 * 2^-152 -> 2^-126" },
        { 0.0f, PAIR(b16(0x1p-63f), b16(0x1p-65f)), PAIR(b16(0x1p-63f), b16(0x1p-65f)), 0x00800000u, "FTZ per FMA: 2^-130 flushed before + 2^-126" },
        { -0.0f, PAIR(z, z), PAIR(one, one), 0x00000000u, "signed zeros: -0 + 0 * 1 + 0 * 1 -> +0" },
        { 0.0f, PAIR(b16(-1.0f), one), PAIR(one, one), 0x00000000u, "exact cancellation: 0 + 1 - 1 -> +0" },
        { 0.0f, PAIR(z, b16(0x1p127f)), PAIR(z, b16(0x1p127f)), 0x7f800000u, "overflow: 2^127 * 2^127 -> inf" },
        { 0x1.fffffep127f, PAIR(z, b16(0x1p52f)), PAIR(z, b16(0x1p52f)), 0x7f800000u, "overflow: FLT_MAX + 2^104 (the tie) -> inf" },
        { 1.0f, PAIR(z, 0x7f80), PAIR(z, z), 0xffc00000u, "inf * 0 -> the default NaN" },
        { fb(0x7f800000u), PAIR(z, 0x7f80), PAIR(z, b16(-1.0f)), 0xffc00000u, "inf - inf -> the default NaN" },
        { 1.0f, PAIR(z, 0x7f81), PAIR(z, one), 0x7fc10000u, "a signalling NaN factor: quieted, propagated" },
        { fb(0x7f800123u), PAIR(one, one), PAIR(one, one), 0x7fc00123u, "a signalling NaN accumulator: quieted, propagated" },
        { fb(0x7fc00123u), PAIR(z, 0x7fc1), PAIR(z, one), 0x7fc10000u, "NaN factor and NaN accumulator: the factor's" },
        { 1.0f, PAIR(z, 0x7fc1), PAIR(z, 0x7fc5), 0x7fc10000u, "NaN in both factors: the first source's (a)" },
        { 1.0f, PAIR(0xffc2, 0x7fc1), PAIR(one, one), 0xffc20000u, "NaN in the odd then the even product: the even factor's" },
    };
    int ne = (int)(sizeof(E) / sizeof(E[0])), bad = 0;
    for (int i = 0; i < ne; i += 16) {
        float c[16], rh[16], rk[16];
        uint32_t a[16], b[16];
        int m = ne - i < 16 ? ne - i : 16;
        for (int l = 0; l < 16; l++) {
            int k = i + (l < m ? l : 0);
            c[l] = E[k].c; a[l] = E[k].a; b[l] = E[k].b;
        }
        if (has_hw) hw16(rh, c, a, b);
        kern16(rk, c, a, b);
        for (int l = 0; l < m; l++) {
            uint32_t w = E[i + l].want, t = bits(twin(c[l], a[l], b[l])), x = bits(ref(c[l], a[l], b[l]));
            int nan = isnan(fb(w)); // (a NaN: the instruction's payload exactly, a NaN for the others)
            int ok = (!has_hw || bits(rh[l]) == w) && (nan ? isnan(rk[l]) && isnan(fb(t)) && isnan(fb(x)) : bits(rk[l]) == w && t == w && x == w);
            snprintf(d, sizeof(d), "%s: want %08x, instruction %08x, att512_dp %08x, twin %08x, reference %08x", E[i + l].what, w,
                     has_hw ? bits(rh[l]) : 0, bits(rk[l]), t, x);
            if (!ok) check(0, "bf16dp/edge", d);
            bad += !ok;
        }
    }
    snprintf(d, sizeof(d), "%d edge cases: the instruction%s, att512_dp, the C twin and the reference give the expected results", ne,
             has_hw ? "" : " (absent)");
    if (!bad) check(1, "bf16dp/edge", d);

    // 2. random lanes
    const char *names[6] = { "all bit patterns", "products cancelling an accumulator ~1", "around 2^-126", "near overflow",
                             "mixed exponents", "denormal accumulators / factors" };
    for (int kind = 0; kind < 6; kind++) {
        long mh = 0, mk = 0, mt = 0, mx = 0, nans = 0, pay = 0;
        for (long it = 0; it < n; it += 16) {
            float c[16], rh[16], rk[16];
            uint32_t a[16], b[16];
            for (int l = 0; l < 16; l++) {
                uint16_t alo, ahi, blo, bhi;
                float cc;
                switch (kind) {
                case 0:
                    alo = (uint16_t)rnd(); ahi = (uint16_t)rnd(); blo = (uint16_t)rnd(); bhi = (uint16_t)rnd(); cc = fb((uint32_t)rnd());
                    break;
                case 1:
                    alo = rbf(120, 134); ahi = rbf(120, 134); blo = rbf(120, 134); bhi = rbf(120, 134);
                    cc = -(b2f(alo) * b2f(blo) + b2f(ahi) * b2f(bhi));
                    cc = fb(bits(cc) ^ (uint32_t)(rnd() % 64));
                    if (rnd() & 1) cc = fb(bits(cc) ^ 0x80000000u);
                    break;
                case 2: { // factor exponents summing to ~-126, an accumulator near 2^-126 (or cancelling the odd product)
                    ahi = rbf(1, 127); alo = rbf(1, 127);
                    int eh = 128 - ((ahi >> 7) & 0xff) + (int)(rnd() % 5) - 2, el = 128 - ((alo >> 7) & 0xff) + (int)(rnd() % 5) - 2;
                    bhi = (uint16_t)((rnd() & 1) << 15 | (unsigned)(eh < 1 ? 1 : eh) << 7 | (rnd() & 0x7f));
                    blo = (uint16_t)((rnd() & 1) << 15 | (unsigned)(el < 1 ? 1 : el) << 7 | (rnd() & 0x7f));
                    cc = fb((uint32_t)(rnd() & 0x80000000u) | (uint32_t)(rnd() % 4) << 23 | (uint32_t)(rnd() & 0x7fffff));
                    if (rnd() % 4 == 0) cc = -(b2f(ahi) * b2f(bhi));
                    break;
                }
                case 3:
                    alo = rbf(190, 254); ahi = rbf(190, 254); blo = rbf(190, 254); bhi = rbf(190, 254);
                    cc = fb((uint32_t)(rnd() & 0x80000000u) | (uint32_t)(250 + rnd() % 5) << 23 | (uint32_t)(rnd() & 0x7fffff));
                    break;
                case 4:
                    alo = rbf(60, 140); ahi = rbf(60, 140); blo = rbf(100, 140); bhi = rbf(100, 140);
                    cc = fb((uint32_t)(rnd() & 0x80000000u) | (uint32_t)(100 + rnd() % 40) << 23 | (uint32_t)(rnd() & 0x7fffff));
                    if (rnd() % 8 == 0) cc = 0.0f;
                    break;
                default:
                    alo = (uint16_t)(rnd() & 0x807f); ahi = rbf(0, 70); blo = rbf(0, 127); bhi = rbf(0, 70);
                    cc = fb((uint32_t)(rnd() & 0x807fffffu));
                    if (rnd() % 3 == 0) cc = fb((uint32_t)(rnd() & 0x80000000u) | (uint32_t)(rnd() % 3) << 23 | (uint32_t)(rnd() & 0x7fffff));
                    break;
                }
                c[l] = cc; a[l] = PAIR(alo, ahi); b[l] = PAIR(blo, bhi);
            }
            if (has_hw) hw16(rh, c, a, b);
            kern16(rk, c, a, b);
            for (int l = 0; l < 16; l++) {
                uint32_t t = bits(twin(c[l], a[l], b[l])), x = bits(ref(c[l], a[l], b[l])), k = bits(rk[l]);
                uint32_t h = has_hw ? bits(rh[l]) : k;
                if (isnan(fb(h))) { // NaN: NaN everywhere (payloads: counted, not required)
                    nans++;
                    mh += !isnan(fb(t)) || !isnan(fb(k)) || !isnan(fb(x));
                    pay += (t != h) + (k != h);
                    continue;
                }
                mk += k != h; mt += t != h; mx += x != h;
            }
        }
        snprintf(d, sizeof(d), "%s: %ld lanes (%ld NaN, %ld other NaN payloads): differ from the %s: att512_dp %ld, C twin %ld, "
                 "reference %ld; NaN-ness %ld", names[kind], n, nans, pay, has_hw ? "instruction" : "twin", mk, mt, mx, mh);
        check(!mh && !mk && !mt && !mx, "bf16dp/random", d);
    }

    // 3. conversions: att512_cvt_bf16 == f32_bf16_daz (all 2^32 inputs), att512_f16_pairs == f32_bf16(f16 as fp32)
    long cbad = 0, cden = 0;
    for (uint64_t u0 = 0; u0 < (1ull << 32); u0 += 16) {
        uint32_t in[16];
        uint16_t out[16];
        for (int l = 0; l < 16; l++) in[l] = (uint32_t)(u0 + l);
        _mm256_storeu_si256((__m256i *)out, att512_cvt_bf16(_mm512_castsi512_ps(_mm512_loadu_si512(in))));
        for (int l = 0; l < 16; l++) {
            cbad += out[l] != f32_bf16_daz(fb(in[l]));
            cden += out[l] != f32_bf16(fb(in[l])); // (the denormals f32_bf16 rounds: why q uses f32_bf16_daz)
        }
    }
    snprintf(d, sizeof(d), "att512_cvt_bf16 vs f32_bf16_daz on all 2^32 inputs: %ld differ (vs f32_bf16: %ld, the denormals)",
             cbad, cden);
    check(!cbad, "bf16dp/cvt", d);
    long pbad = 0;
    for (int h0 = 0; h0 < 65536; h0 += 16) {
        uint16_t va[16], vb[16];
        uint32_t o[16];
        for (int l = 0; l < 16; l++) { va[l] = (uint16_t)(h0 + l); vb[l] = (uint16_t)(65535 - h0 - l); }
        _mm512_storeu_si512(o, att512_f16_pairs(va, vb));
        for (int l = 0; l < 16; l++)
            pbad += o[l] != PAIR(f32_bf16(_cvtsh_ss(va[l])), f32_bf16(_cvtsh_ss(vb[l])));
    }
    snprintf(d, sizeof(d), "att512_f16_pairs vs f32_bf16(f16 -> fp32), all 2^16 f16 values in both halves: %ld differ", pbad);
    check(!pbad, "bf16dp/f16_pairs", d);

    // the twin needs FTZ / DAZ (forward.c sets them for the attention): without, it is IEEE's gradual underflow
    _mm_setcsr(csr0 & ~0x8040u);
    long off = 0;
    for (long it = 0; it < 4096; it++) {
        float c = fb((uint32_t)(rnd() & 0x807fffffu));
        uint32_t a = PAIR(rbf(0, 70), rbf(0, 70)), b = PAIR(rbf(0, 127), rbf(0, 70));
        off += bits(twin(c, a, b)) != bits(ref(c, a, b));
    }
    _mm_setcsr(csr0);
    printf("# (without FTZ / DAZ the C twin differs on %ld of 4096 lanes with denormal inputs: why the attention sets them)\n", off);
    return fails ? 1 : 0;
}
#endif
