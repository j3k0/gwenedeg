/* Sampler. */
#include "internal.h"

void softmax(float *x, int size) {
    // find max value (for numerical stability)
    float max_val = 0;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    // exp and sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // normalize
    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

int sample_argmax(float *probabilities, int n) {
    // return the index that has the highest probability: the first one holding the maximum (as the
    // original scalar loop with a strict >). Two passes that vectorize, the maximum (exact in any
    // order) and the search for its first occurrence: ~125 -> ~20 us per token on 151,936 logits
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++)
        max_p = probabilities[i] > max_p ? probabilities[i] : max_p;
    int i = 0;
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
    __m512 m = _mm512_set1_ps(max_p);
    for (; i + 16 <= n; i += 16) {
        __mmask16 k = _mm512_cmp_ps_mask(_mm512_loadu_ps(probabilities + i), m, _CMP_EQ_OQ);
        if (k)
            return i + __builtin_ctz(k);
    }
#endif
    for (; i < n; i++)
        if (probabilities[i] == max_p)
            return i;
    return 0;
}

int sample_mult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1; // in case of rounding errors
}

static inline int topp_before(const ProbIndex *a, const ProbIndex *b) {
    // nucleus order: higher probability first, lower index first on ties. (The old code sorted the
    // candidates, collected in index order, with qsort on the probability only: glibc's qsort is a
    // stable merge sort at these sizes (checked on glibc 2.41), so ties came out by index too)
    return a->prob > b->prob || (a->prob == b->prob && a->index < b->index);
}

static void topp_sift_down(ProbIndex *h, int n, int i) {
    // max-heap (topp_before order) of h[0..n): restore the heap property below i
    ProbIndex x = h[i];
    for (;;) {
        int l = 2 * i + 1;
        if (l >= n) break;
        int m = l + 1 < n && topp_before(&h[l + 1], &h[l]) ? l + 1 : l;
        if (!topp_before(&h[m], &x)) break;
        h[i] = h[m];
        i = m;
    }
    h[i] = x;
}

static int topp_candidates(const float *p, int n, float cutoff, ProbIndex *out) {
    // the (prob, index) of every p[i] >= cutoff, in index order; returns their count. They are few
    // (median ~40 of 151,936 at -t 0.8 -p 0.9), so the scan looks for blocks holding any
    int n0 = 0, i = 0;
#if defined(__AVX512F__) && !defined(RUNQ_GENERIC)
    __m512 c = _mm512_set1_ps(cutoff);
    for (; i + 16 <= n; i += 16) // one compare per 16 values, then the set bits
        for (__mmask16 m = _mm512_cmp_ps_mask(_mm512_loadu_ps(p + i), c, _CMP_GE_OQ); m; m &= m - 1) {
            int j = i + __builtin_ctz(m);
            out[n0].index = j;
            out[n0].prob = p[j];
            n0++;
        }
#else
    for (; i + 64 <= n; i += 64) { // blocks of 64: their max in a loop gcc vectorizes, then a scan
        float m = p[i];
        for (int j = 1; j < 64; j++)
            m = p[i + j] > m ? p[i + j] : m;
        if (!(m >= cutoff))
            continue;
        for (int j = i; j < i + 64; j++)
            if (p[j] >= cutoff) {
                out[n0].index = j;
                out[n0].prob = p[j];
                n0++;
            }
    }
#endif
    for (; i < n; i++)
        if (p[i] >= cutoff) {
            out[n0].index = i;
            out[n0].prob = p[i];
            n0++;
        }
    return n0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()
    // Values smaller than (1 - topp) / (n - 1) cannot be part of the result, so only the others
    // are candidates. They are not sorted (was: qsort, ~5 ms at -t 2 with ~44k candidates): a
    // max-heap of them is popped only until the nucleus is complete (heapify O(n0), then
    // O(log n0) per nucleus token). The same order (topp_before), the same sums in the same order,
    // so the same token for the same coin as the old full sort.
    const float cutoff = (1.0f - topp) / (n - 1);
    int n0 = topp_candidates(probabilities, n, cutoff, probindex);
    for (int i = n0 / 2 - 1; i >= 0; i--)
        topp_sift_down(probindex, n0, i);

    // pop the candidates in order (the i-th one lands at probindex[n0 - 1 - i]) until the
    // cumulative probability exceeds topp
    float cumulative_prob = 0;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        int h = n0 - i; // heap size
        ProbIndex top = probindex[0];
        probindex[0] = probindex[h - 1];
        probindex[h - 1] = top;
        topp_sift_down(probindex, h - 1, 0);
        cumulative_prob += top.prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[n0 - 1 - i].prob;
        if (r < cdf)
            return probindex[n0 - 1 - i].index;
    }
    return probindex[n0 - 1 - last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    // sample the token given the logits and some hyperparameters
    if (sampler->temperature == 0) {
        // greedy argmax sampling: take the token with the highest probability
        return sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            return sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
}

