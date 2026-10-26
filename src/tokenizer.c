/* Tokenizer. */
#include "internal.h"

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens
//
// cwayn rewrite, matching HF tokenizers on Qwen2/Qwen3's tokenizer.json:
//   1. added tokens (<|im_start|>, <think>, ...) are split out first, leftmost-longest match
//   2. normalizer (NFC), on the text between added tokens (nfc_check / nfc_normalize, tables in
//      unicode_nfc.h: Unicode 15.1 data with the overrides that match HF tokenizers' older data)
//   3. pre-tokenizer: Qwen2's split regex, hand-written in pretok_next(), with the Unicode
//      classes \p{L}, \p{N}, \s from unicode_tables.h (tools/gen_unicode_tables.py)
//   4. byte-level BPE inside each chunk, starting from the single-byte tokens
// Merge order: the .tokenizer file (third_party/qwen3.c/export.py) has one score per token,
// -log(rank + 1) of the merge producing it, and no merge pairs. bpe_chunk() merges any adjacent
// pair whose concatenation is a vocab token, best score first, leftmost on ties (tiktoken
// style). Verified against HF's BPE with tokenizer.json's model.merges: identical on all
// 151,643 vocab strings and 300k random token concatenations (no pre-tokenizer), since Qwen's
// merge list is the tiktoken-consistent one: one merge per token, in id order, each the split
// tiktoken-style BPE itself produces from the lower-ranked tokens.

#include "unicode_tables.h"
#include "unicode_nfc.h"

// tokens[] capacity encode() needs for a text of n bytes: one token per byte of the NFC-normalized
// text, which is at most 3x as long (e.g. U+1D160: 4 bytes -> 3 code points, 12 bytes)
#define ENCODE_MAX_TOKENS(n) (3 * (size_t)(n) + 3)

void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

static inline uint64_t bytes_hash(const unsigned char *s, int n) {
    uint64_t h = 1469598103934665603ull; // FNV-1a, then a murmur3 finalizer for the low bits
    for (int i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ull; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33;
    return h;
}

void tokenizer_index(Tokenizer *t);

int vocab_lookup(Tokenizer *t, const unsigned char *s, int n) {
    // id of the regular token with exactly these n bytes, or -1
    uint64_t h = bytes_hash(s, n);
    uint32_t tag = (uint32_t)(h >> 32);
    for (uint32_t i = (uint32_t)h & t->hash_mask;; i = (i + 1) & t->hash_mask) {
        int id = t->hash_ids[i];
        if (id < 0) return -1;
        if (t->hash_tags[i] == tag && t->vocab_len[id] == n && !memcmp(t->vocab[id], s, n)) return id;
    }
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    // read the whole file: header (max_token_length, bos, eos), then per token
    // float score, int32 length, bytes; tokens missing at the end (padding ids) are empty
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fseek(file, 0, SEEK_END);
    long fsize = ftell(file);
    fseek(file, 0, SEEK_SET);
    unsigned char *buf = malloc(fsize > 0 ? fsize : 1);
    if (fsize < 12 || fread(buf, 1, fsize, file) != (size_t)fsize) {
        fprintf(stderr, "Couldn't read tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE);
    }
    fclose(file);
    memcpy(&t->max_token_length, buf, 4);
    memcpy(&t->bos_token_id, buf + 4, 4);
    memcpy(&t->eos_token_id, buf + 8, 4);

    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_len = malloc(vocab_size * sizeof(int));
    t->merge_scores = malloc(vocab_size * sizeof(float));
    t->vocab_data = malloc(fsize + vocab_size); // string bytes + one NUL per token
    if (!t->vocab || !t->vocab_len || !t->merge_scores || !t->vocab_data) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    long off = 12;
    char *dst = t->vocab_data;
    unsigned int max_len = 0;
    for (int i = 0; i < vocab_size; i++) {
        int32_t len = 0;
        float score = -1e6f;
        if (off + 8 <= fsize) {
            memcpy(&score, buf + off, 4);
            memcpy(&len, buf + off + 4, 4);
            if (len < 0 || off + 8 + len > fsize) { fprintf(stderr, "Corrupt tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
            memcpy(dst, buf + off + 8, len);
            off += 8 + len;
        }
        t->vocab[i] = dst;
        t->vocab_len[i] = len;
        t->merge_scores[i] = score;
        dst[len] = 0;
        dst += len + 1;
        if ((unsigned int)len > max_len) max_len = len;
    }
    free(buf);
    t->max_token_length = max_len;
    tokenizer_index(t);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void tokenizer_index(Tokenizer *t) {
    // classify: single bytes, merge results (score > -1e6) and added tokens (the rest, len > 1;
    // export.py gives every token that no merge produces the score -1e6)
    int vocab_size = t->vocab_size;
    for (int b = 0; b < 256; b++) t->byte_token[b] = -1;
    uint32_t cap = 1;
    while (cap < 2u * (uint32_t)vocab_size) cap <<= 1;
    t->hash_mask = cap - 1;
    t->hash_ids = malloc(cap * sizeof(int32_t));
    t->hash_tags = malloc(cap * sizeof(uint32_t));
    t->added = malloc(vocab_size * sizeof(int));
    if (!t->hash_ids || !t->hash_tags || !t->added) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    memset(t->hash_ids, 0xff, cap * sizeof(int32_t));
    t->n_added = 0;
    memset(t->added_first, 0, sizeof(t->added_first));
    for (int i = 0; i < vocab_size; i++) {
        int len = t->vocab_len[i];
        const unsigned char *s = (const unsigned char *)t->vocab[i];
        if (len == 0) continue;
        if (len > 1 && t->merge_scores[i] <= -1e6f) {
            t->added[t->n_added++] = i;
            t->added_first[s[0]] = 1;
            continue;
        }
        if (len == 1 && t->byte_token[s[0]] < 0) t->byte_token[s[0]] = i;
        uint64_t h = bytes_hash(s, len);
        uint32_t j = (uint32_t)h & t->hash_mask;
        int dup = 0;
        for (; t->hash_ids[j] >= 0; j = (j + 1) & t->hash_mask) {
            int o = t->hash_ids[j];
            if (t->vocab_len[o] == len && !memcmp(t->vocab[o], s, len)) { dup = 1; break; } // keep the first
        }
        if (dup) continue;
        t->hash_ids[j] = i;
        t->hash_tags[j] = (uint32_t)(h >> 32);
    }
    for (int b = 0; b < 256; b++) {
        if (t->byte_token[b] < 0) { fprintf(stderr, "Tokenizer has no token for byte 0x%02x (not byte-level BPE?)\n", b); exit(EXIT_FAILURE); }
    }
    for (int i = 1; i < t->n_added; i++) { // longest first (insertion sort, ~26 entries), stable
        int id = t->added[i], k = i;
        for (; k > 0 && t->vocab_len[t->added[k - 1]] < t->vocab_len[id]; k--) t->added[k] = t->added[k - 1];
        t->added[k] = id;
    }
}

// Qwen3's chat template (tokenizer_config.json) rendered for one user message "%s", with and without
// a system message "%s" and enable_thinking: the strings export.py writes into the .template* files
// (byte-identical; make test compares the chat token ids of a .bin and a GGUF run). Used for GGUF
// files, whose tokenizer.chat_template is Jinja (not interpreted here)
#define QWEN3_TMPL_SYS "<|im_start|>system\n%s<|im_end|>\n"
#define QWEN3_TMPL_USER "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n"
#define QWEN3_TMPL_NOTHINK "<think>\n\n</think>\n\n"

static int gpt2_byte(uint32_t cp) {
    // GPT-2's bytes_to_unicode, inverted: the printable bytes ('!'..'~', 0xa1..0xac, 0xae..0xff)
    // stand for themselves, the other 68 (0x00..0x20, 0x7f..0xa0, 0xad) for U+0100.. in byte order;
    // -1 if cp is neither
    if ((cp >= 0x21 && cp <= 0x7e) || (cp >= 0xa1 && cp <= 0xac) || (cp >= 0xae && cp <= 0xff)) return (int)cp;
    if (cp < 0x100 || cp >= 0x100 + 68) return -1;
    int n = (int)cp - 0x100;
    return n <= 0x20 ? n : n < 33 + 34 ? 0x7f + (n - 33) : 0xad;
}

static inline uint32_t utf8_cp(const unsigned char *p, const unsigned char *end, int *len);

static int gpt2_decode(const char *s, uint64_t n, char *out) {
    // a token string in GPT-2's byte-level form -> its raw bytes (at most n); -1 if it is not one
    const unsigned char *p = (const unsigned char *)s, *end = p + n;
    int len = 0;
    while (p < end) {
        if (*p >= 0x21 && *p <= 0x7e) { out[len++] = (char)*p++; continue; } // printable ASCII: itself
        int cl, b = gpt2_byte(utf8_cp(p, end, &cl)); // (invalid UTF-8: U+FFFD, not a byte)
        if (b < 0) return -1;
        out[len++] = (char)b;
        p += cl;
    }
    return len;
}

typedef struct { Tokenizer *t; const Gguf *g; const uint64_t *mo; int *mid; long nmg; unsigned int max_len; long next; } MergeJob;

static void merge_worker(void *arg, int tid, int nt) {
    // build_tokenizer_gguf: the tokens produced by the merges (mid[r]: id, -1 none, -2 not GPT-2
    // form), in claims of 4,096 merges
    (void)tid; (void)nt;
    MergeJob *M = arg;
    char *buf = malloc(2 * (size_t)M->max_len + 2);
    if (!buf) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    for (long r0; (r0 = __atomic_fetch_add(&M->next, 4096, __ATOMIC_RELAXED)) < M->nmg;)
        for (long r = r0; r < r0 + 4096 && r < M->nmg; r++) {
            uint64_t sl, p = M->mo[r];
            memcpy(&sl, M->g->map + p, 8); // (in the file: checked by the walk above)
            const char *s = (const char *)M->g->map + p + 8, *sp;
            M->mid[r] = -1;
            if (sl > 2 * (uint64_t)M->max_len + 1 || !(sp = memchr(s, ' ', sl))) continue;
            int la = gpt2_decode(s, sp - s, buf), lb = la < 0 ? -1 : gpt2_decode(sp + 1, sl - (sp + 1 - s), buf + la);
            M->mid[r] = la < 0 || lb < 0 ? -2 : vocab_lookup(M->t, (const unsigned char *)buf, la + lb);
        }
    free(buf);
}

void build_tokenizer_gguf(Tokenizer *t, const Gguf *g, int vocab_size, int enable_thinking) {
    // The tokenizer from GGUF metadata (llama.cpp's "gpt2" model with the "qwen2" pre-tokenizer,
    // which is what encode() implements):
    //   tokenizer.ggml.tokens: token strings, in GPT-2's byte-to-unicode form for normal tokens
    //     (decoded to raw bytes, as the .tokenizer file stores them), as is for added ones
    //   tokenizer.ggml.token_type: 1 normal, 3 control and 4 user-defined = added tokens (HF's
    //     added_tokens, matched in the raw text), 5 unused (llama.cpp's [PAD...] fillers up to the
    //     embedding size: empty here, as the ids missing from the .tokenizer file)
    //   tokenizer.ggml.merges: "left right" in rank order. Score of the token a merge of rank r
    //     produces: -r (the .tokenizer file has -log(r + 1): the same order, so the same BPE);
    //     the first merge counts if several give one token; a normal token of several bytes that
    //     no merge produces gets -999999 (after all merges; Qwen3 has none). Checked: the tokens
    //     decode to exactly the .tokenizer file's bytes and the merges equal tokenizer.json's
    //   eos = tokenizer.ggml.eos_token_id; bos (generation also stops there) =
    //     tokenizer.ggml.bos_token_id, else the token <|endoftext|> (Qwen's, as config.json), else eos
    uint64_t n;
    const char *model = gguf_get_str(g, "tokenizer.ggml.model", &n);
    if (!model || n != 4 || memcmp(model, "gpt2", 4)) gguf_fail(g, "tokenizer.ggml.model must be gpt2 (byte-level BPE)%s", NULL);
    const char *pre = gguf_get_str(g, "tokenizer.ggml.pre", &n);
    if (!pre || n != 5 || memcmp(pre, "qwen2", 5)) gguf_fail(g, "tokenizer.ggml.pre must be qwen2 (the pre-tokenizer runq implements)%s", NULL);
    const GgufKV *tk = gguf_get_array(g, "tokenizer.ggml.tokens", GV_STR);
    const GgufKV *ty = gguf_get_array(g, "tokenizer.ggml.token_type", GV_I32);
    const GgufKV *mg = gguf_get_array(g, "tokenizer.ggml.merges", GV_STR);
    if (!tk || !mg) gguf_fail(g, "tokenizer.ggml.tokens / merges missing%s", NULL);
    if (tk->n > (uint64_t)vocab_size) gguf_fail(g, "more tokens than embedding rows%s", NULL);
    if (ty && ty->n != tk->n) gguf_fail(g, "tokenizer.ggml.token_type has the wrong length%s", NULL);

    t->vocab_size = vocab_size;
    t->vocab = malloc(vocab_size * sizeof(char *));
    t->vocab_len = malloc(vocab_size * sizeof(int));
    t->merge_scores = malloc(vocab_size * sizeof(float));
    uint64_t pos = tk->off, bytes = 0;
    for (uint64_t i = 0; i < tk->n; i++) { // (bounds checked by gguf_parse, again here)
        const char *s;
        bytes += gguf_str(g, &pos, &s, "tokenizer.ggml.tokens");
    }
    t->vocab_data = malloc(bytes + vocab_size);
    if (!t->vocab || !t->vocab_len || !t->merge_scores || !t->vocab_data) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    char *dst = t->vocab_data;
    unsigned int max_len = 0;
    pos = tk->off;
    for (int i = 0; i < vocab_size; i++) {
        const char *s = "";
        uint64_t sl = 0;
        int32_t type = 1, len = 0;
        if ((uint64_t)i < tk->n) {
            sl = gguf_str(g, &pos, &s, "tokenizer.ggml.tokens");
            if (sl > 65536) gguf_fail(g, "a token string is too long%s", NULL);
            if (ty) memcpy(&type, g->map + ty->off + 4 * (uint64_t)i, 4);
        } else {
            type = 5;
        }
        if (type == 5) {
            len = 0;
        } else if (type == 3 || type == 4) { // added: raw text
            len = (int)sl;
            memcpy(dst, s, sl);
        } else if ((len = gpt2_decode(s, sl, dst)) < 0) {
            char id[24];
            snprintf(id, sizeof(id), "%d", i);
            gguf_fail(g, "token %s is not in GPT-2 byte-level form", id);
        }
        t->vocab[i] = dst;
        t->vocab_len[i] = len;
        t->merge_scores[i] = (type == 3 || type == 4) && len > 1 ? -1e6f : -999999.0f;
        dst[len] = 0;
        dst += len + 1;
        if ((unsigned int)len > max_len) max_len = len;
    }
    t->max_token_length = max_len;
    tokenizer_index(t);

    // merge ranks -> scores of the tokens they produce: the merge strings located (serial walk),
    // decoded and looked up in parallel (151k hash lookups, ~12 ms on one thread), then the scores
    // set in rank order. A GPT-2 form character is 1-2 bytes, so a merge string longer than
    // 2 * max_len + 1 cannot give a token (and the decoded bytes fit buf)
    long nmg = (long)mg->n;
    uint64_t *mo = malloc((nmg + 1) * sizeof(uint64_t));
    int *mid = malloc((nmg + 1) * sizeof(int));
    if (!mo || !mid) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    pos = mg->off;
    for (long r = 0; r < nmg; r++) {
        const char *s;
        mo[r] = pos;
        gguf_str(g, &pos, &s, "tokenizer.ggml.merges");
    }
    MergeJob mj = { t, g, mo, mid, nmg, max_len, 0 };
    rq_parallel(merge_worker, &mj);
    for (long r = 0; r < nmg; r++) {
        if (mid[r] == -2) gguf_fail(g, "a merge is not in GPT-2 byte-level form%s", NULL);
        if (mid[r] >= 0 && t->merge_scores[mid[r]] == -999999.0f)
            t->merge_scores[mid[r]] = -(float)r;
    }
    free(mo);
    free(mid);

    int64_t v;
    if (!gguf_get_int(g, "tokenizer.ggml.eos_token_id", &v) || v < 0 || v >= vocab_size)
        gguf_fail(g, "tokenizer.ggml.eos_token_id missing or out of range%s", NULL);
    t->eos_token_id = (unsigned int)v;
    t->bos_token_id = t->eos_token_id;
    if (gguf_get_int(g, "tokenizer.ggml.bos_token_id", &v)) {
        if (v < 0 || v >= vocab_size) gguf_fail(g, "tokenizer.ggml.bos_token_id out of range%s", NULL);
        t->bos_token_id = (unsigned int)v;
    } else {
        for (int k = 0; k < t->n_added; k++)
            if (t->vocab_len[t->added[k]] == 13 && !memcmp(t->vocab[t->added[k]], "<|endoftext|>", 13))
                t->bos_token_id = t->added[k];
    }

    strcpy(t->prompt_template, enable_thinking ? QWEN3_TMPL_USER : QWEN3_TMPL_USER QWEN3_TMPL_NOTHINK);
    strcpy(t->system_prompt_template, enable_thinking ? QWEN3_TMPL_SYS QWEN3_TMPL_USER : QWEN3_TMPL_SYS QWEN3_TMPL_USER QWEN3_TMPL_NOTHINK);
}

void free_tokenizer(Tokenizer *t) {
    free(t->vocab);
    free(t->vocab_len);
    free(t->vocab_data);
    free(t->merge_scores);
    free(t->hash_ids);
    free(t->hash_tags);
    free(t->added);
}

char *decode(Tokenizer *t, int token) {
    // raw token bytes; a code point split over several tokens comes out whole once all are printed
    return t->vocab[token];
}

// --- pre-tokenizer: Qwen2's Split regex (tokenizer.json, behavior Isolated)
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

typedef struct { uint32_t cp; int len; int cls; } UChar;

static inline int ucls(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp | 0x20) >= 'a' && (cp | 0x20) <= 'z') return UCLS_L;
        if (cp >= '0' && cp <= '9') return UCLS_N;
        if (cp == ' ' || (cp >= '\t' && cp <= '\r')) return UCLS_WS;
        return UCLS_OTHER;
    }
    int lo = 0, hi = UCLS_NRANGES - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (cp < ucls_lo[mid]) hi = mid - 1;
        else if (cp > (ucls_hi_cls[mid] & 0xffffff)) lo = mid + 1;
        else return ucls_hi_cls[mid] >> 24;
    }
    return UCLS_OTHER;
}

static inline uint32_t utf8_cp(const unsigned char *p, const unsigned char *end, int *len) {
    // decode one UTF-8 code point; an invalid byte gives U+FFFD with *len = 1
    unsigned char b = p[0];
    *len = 1;
    if (b < 0x80) return b;
    int n = b >= 0xf0 ? (b <= 0xf4 ? 4 : 0) : b >= 0xe0 ? 3 : b >= 0xc2 ? 2 : 0;
    if (n == 0 || end - p < n) return 0xfffd;
    uint32_t cp = b & (0x7f >> n);
    for (int i = 1; i < n; i++) {
        if ((p[i] & 0xc0) != 0x80) return 0xfffd;
        cp = (cp << 6) | (p[i] & 0x3f);
    }
    if ((n == 3 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) || (n == 4 && (cp < 0x10000 || cp > 0x10ffff)))
        return 0xfffd;
    *len = n;
    return cp;
}

static inline UChar uchar_at(const unsigned char *p, const unsigned char *end) {
    // decode one UTF-8 code point; an invalid byte counts as one "other" character (U+FFFD is one)
    UChar c;
    c.cp = utf8_cp(p, end, &c.len);
    c.cls = ucls(c.cp);
    return c;
}

static inline const unsigned char *skip_class(const unsigned char *p, const unsigned char *end, int cls) {
    while (p < end) {
        UChar c = uchar_at(p, end);
        if (c.cls != cls) break;
        p += c.len;
    }
    return p;
}

static inline int ascii_lower(int c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; }

static int contraction_len(const unsigned char *p, const unsigned char *end) {
    // bytes matched by (?i:s|t|re|ve|m|ll|d) after the apostrophe, 0 if none. Case folding is
    // Unicode's: U+017F LATIN SMALL LETTER LONG S also matches 's' (checked against HF)
    if (p >= end) return 0;
    int c0 = ascii_lower(p[0]), c1 = p + 1 < end ? ascii_lower(p[1]) : 0;
    if (c0 == 's' || c0 == 't' || c0 == 'm' || c0 == 'd') return 1;
    if (p[0] == 0xc5 && c1 == 0xbf) return 2;
    if (((c0 == 'r' || c0 == 'v') && c1 == 'e') || (c0 == 'l' && c1 == 'l')) return 2;
    return 0;
}

static const unsigned char *pretok_next(const unsigned char *p, const unsigned char *end) {
    // end of the regex match starting at p (p < end); every character matches some alternative
    UChar c = uchar_at(p, end);
    const unsigned char *q = p + c.len;
    int crlf = c.cp == '\r' || c.cp == '\n';
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (c.cp == '\'') {
        int n = contraction_len(q, end);
        if (n) return q + n;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+
    if (c.cls == UCLS_L) return skip_class(q, end, UCLS_L);
    if (c.cls != UCLS_N && !crlf && q < end && uchar_at(q, end).cls == UCLS_L) return skip_class(q, end, UCLS_L);
    // \p{N}: one digit at a time
    if (c.cls == UCLS_N) return q;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*' (only ASCII space as the optional prefix)
    const unsigned char *r = c.cp == ' ' ? q : p;
    if (r < end && uchar_at(r, end).cls == UCLS_OTHER) {
        r = skip_class(r, end, UCLS_OTHER);
        while (r < end && (*r == '\r' || *r == '\n')) r++;
        return r;
    }
    // c is whitespace. \s*[\r\n]+ : the whitespace run up to its last CR/LF, if any;
    // else \s+(?!\S) : the run minus its last character if a non-space follows (that last
    // space then starts the next chunk, e.g. " word"); else \s+ : the run (1 char, or at the end)
    const unsigned char *last_nl = NULL, *last_start = p;
    r = p;
    while (r < end) {
        UChar w = uchar_at(r, end);
        if (w.cls != UCLS_WS) break;
        if (w.cp == '\r' || w.cp == '\n') last_nl = r + w.len;
        last_start = r;
        r += w.len;
    }
    if (last_nl) return last_nl;
    if (r == end || last_start == p) return r;
    return last_start;
}

// --- normalizer: NFC (tokenizer.json). All added tokens are "normalized": false, so HF matches them
// on the raw text and normalizes each piece between them on its own: so does encode_segment().
// Invalid UTF-8 (no HF equivalent: a Python str cannot hold it) passes through unchanged, each
// invalid byte acting as U+FFFD would (a starter that composes with nothing): no reordering or
// composition across it, and the valid text around it is normalized as usual.

enum { NFC_YES = 0, NFC_MAYBE = 1, NFC_NO = 2 };
enum { HG_S = 0xAC00, HG_L = 0x1100, HG_V = 0x1161, HG_T = 0x11A7, HG_LN = 19, HG_VN = 21, HG_TN = 28, HG_SN = 11172 };
#define NFC_RAW 0x110000u // + byte: an invalid UTF-8 byte in the code point buffer (in no table)

static inline int nfc_find(uint32_t cp, const uint32_t *blk) {
    // index of cp in nfc_cp[], or -1; blk = nfc_any_blk, or nfc_qc_blk when only ccc / qc matter
    if (cp < NFC_MIN_CP || cp > 0x10ffff || !(blk[cp >> 11] >> (cp >> 6 & 31) & 1)) return -1;
    int lo = 0, hi = NFC_N - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint32_t m = nfc_cp[mid] & 0x1fffff;
        if (cp < m) hi = mid - 1;
        else if (cp > m) lo = mid + 1;
        else return mid;
    }
    return -1;
}

static inline int nfc_info(int k) { // ccc | 256 if NFC_QC = Maybe (can combine with what precedes it)
    return k < 0 ? 0 : (int)(nfc_cp[k] >> 23) | ((nfc_cp[k] >> 21 & 3) == NFC_MAYBE) << 8;
}

static const unsigned char *nfc_check(const unsigned char *p, const unsigned char *end) {
    // quick check: end if the text is NFC (every code point NFC_QC = Yes, combining marks in
    // canonical order), else where normalizing must start: the last code point before the first
    // failure with ccc 0 and NFC_QC = Yes (nothing before it interacts with what follows)
    const unsigned char *safe = p;
    int last = 0;
    while (p < end) {
        if (*p < 0x80) { // ASCII: ccc 0, Yes
            while (++p < end && *p < 0x80) {}
            safe = p - 1;
            last = 0;
            continue;
        }
        int n, ccc = 0;
        int k = nfc_find(utf8_cp(p, end, &n), nfc_qc_blk);
        if (k >= 0) {
            ccc = nfc_cp[k] >> 23;
            if ((nfc_cp[k] >> 21 & 3) != NFC_YES || (ccc && ccc < last)) return safe;
        }
        if (!ccc) safe = p;
        last = ccc;
        p += n;
    }
    return end;
}

static uint32_t nfc_compose(uint32_t a, uint32_t b) {
    // primary composite of the pair, or 0 (Hangul algorithmically; NFC_RAW values are in no key)
    if (a - HG_L < HG_LN && b - HG_V < HG_VN) return HG_S + ((a - HG_L) * HG_VN + (b - HG_V)) * HG_TN;
    if (a - HG_S < HG_SN && (a - HG_S) % HG_TN == 0 && b - HG_T - 1 < HG_TN - 1) return a + (b - HG_T);
    uint64_t key = (uint64_t)a << 21 | b;
    int lo = 0, hi = NFC_NCOMP - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        uint64_t m = nfc_comp[mid] >> 21;
        if (key < m) hi = mid - 1;
        else if (key > m) lo = mid + 1;
        else return (uint32_t)(nfc_comp[mid] & 0x1fffff);
    }
    return 0;
}

static void nfc_sort_marks(uint32_t *cp, uint16_t *inf, int n, uint32_t *tcp, uint16_t *tinf) {
    // stable sort of a run of combining marks by ccc: insertion sort, counting sort for long runs
    if (n < 32) {
        for (int i = 1; i < n; i++) {
            uint32_t c = cp[i];
            uint16_t f = inf[i];
            int j = i;
            for (; j > 0 && (inf[j - 1] & 0xff) > (f & 0xff); j--) { cp[j] = cp[j - 1]; inf[j] = inf[j - 1]; }
            cp[j] = c;
            inf[j] = f;
        }
        return;
    }
    int cnt[257] = {0};
    for (int i = 0; i < n; i++) cnt[(inf[i] & 0xff) + 1]++;
    for (int i = 1; i < 257; i++) cnt[i] += cnt[i - 1];
    for (int i = 0; i < n; i++) { int j = cnt[inf[i] & 0xff]++; tcp[j] = cp[i]; tinf[j] = inf[i]; }
    memcpy(cp, tcp, n * sizeof(uint32_t));
    memcpy(inf, tinf, n * sizeof(uint16_t));
}

static unsigned char *nfc_normalize(const unsigned char *p, const unsigned char *q, const unsigned char *end, int *out_len) {
    // NFC of [p, end) as a new buffer: [p, q) copied (nfc_check), [q, end) decomposed, its marks
    // put in canonical order and composed. Output <= 3x the bytes, <= 2 code points per byte
    // (both asserted by the table generator)
    int nb = (int)(end - q), cap = 2 * nb + 1;
    uint32_t *cp = malloc(2 * (size_t)cap * sizeof(uint32_t)); // code points, then sort scratch
    uint16_t *inf = malloc(2 * (size_t)cap * sizeof(uint16_t)); // nfc_info of each
    unsigned char *out = malloc((q - p) + 3 * (size_t)nb + 1);
    if (!cp || !inf || !out) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
    int n = 0;
    for (const unsigned char *r = q; r < end;) { // decompose
        int len;
        uint32_t c = utf8_cp(r, end, &len);
        if (c == 0xfffd && len == 1) c = NFC_RAW + *r;
        r += len;
        if (c - HG_S < HG_SN) {
            uint32_t s = c - HG_S;
            cp[n] = HG_L + s / (HG_VN * HG_TN); inf[n++] = 0;
            cp[n] = HG_V + s % (HG_VN * HG_TN) / HG_TN; inf[n++] = 256;
            if (s % HG_TN) { cp[n] = HG_T + s % HG_TN; inf[n++] = 256; }
            continue;
        }
        int k = nfc_find(c, nfc_any_blk);
        if (k >= 0 && nfc_dec[k]) {
            const uint32_t *d = nfc_pool + (nfc_dec[k] >> 3);
            for (int i = 0; i < (nfc_dec[k] & 7); i++) { cp[n] = d[i]; inf[n++] = nfc_info(nfc_find(d[i], nfc_qc_blk)); }
        } else {
            cp[n] = c; inf[n++] = nfc_info(k);
        }
    }
    for (int i = 0; i < n;) { // canonical order: each run of combining marks sorted by ccc
        if (!(inf[i] & 0xff)) { i++; continue; }
        int j = i;
        while (j < n && (inf[j] & 0xff)) j++;
        if (j - i > 1) nfc_sort_marks(cp + i, inf + i, j - i, cp + cap, inf + cap);
        i = j;
    }
    int w = 0, starter = -1, last = 0; // compose: last = ccc of the last code point kept
    for (int i = 0; i < n; i++) {
        int cc = inf[i] & 0xff;
        // not blocked: adjacent to the starter, or every mark in between has a lower ccc
        if (starter >= 0 && (inf[i] & 256) && (w == starter + 1 || last < cc)) {
            uint32_t x = nfc_compose(cp[starter], cp[i]);
            if (x) { cp[starter] = x; continue; }
        }
        if (!cc) starter = w;
        last = cc;
        cp[w++] = cp[i];
    }
    unsigned char *o = out;
    memcpy(o, p, q - p);
    o += q - p;
    for (int i = 0; i < w; i++) {
        uint32_t c = cp[i];
        if (c >= NFC_RAW) *o++ = (unsigned char)(c - NFC_RAW);
        else if (c < 0x80) *o++ = c;
        else if (c < 0x800) { *o++ = 0xc0 | c >> 6; *o++ = 0x80 | (c & 0x3f); }
        else if (c < 0x10000) { *o++ = 0xe0 | c >> 12; *o++ = 0x80 | (c >> 6 & 0x3f); *o++ = 0x80 | (c & 0x3f); }
        else { *o++ = 0xf0 | c >> 18; *o++ = 0x80 | (c >> 12 & 0x3f); *o++ = 0x80 | (c >> 6 & 0x3f); *o++ = 0x80 | (c & 0x3f); }
    }
    free(cp);
    free(inf);
    *out_len = (int)(o - out);
    return out;
}

// --- byte-level BPE inside one pre-tokenizer chunk

typedef struct { float score; int pos, len, id; } BpeCand; // merge candidate: symbol at pos + next one

typedef struct {
    int *len, *tok, *prev; // per byte offset: symbol length (0 = not a symbol start), token id, previous symbol
    BpeCand *heap;
    int cap;
} BpeScratch;

static void bpe_reserve(BpeScratch *s, int n) {
    // scratch for chunks of up to n bytes
    if (n < 1) n = 1;
    if (s->len && n <= s->cap) return;
    free(s->len);
    free(s->heap);
    s->cap = n;
    s->len = malloc(3 * (size_t)n * sizeof(int));
    s->tok = s->len + n;
    s->prev = s->tok + n;
    s->heap = malloc(3 * (size_t)n * sizeof(BpeCand)); // n-1 initial pairs + 2 per merge
    if (!s->len || !s->heap) { fprintf(stderr, "malloc failed!\n"); exit(EXIT_FAILURE); }
}

static inline int cand_before(const BpeCand *a, const BpeCand *b) {
    return a->score > b->score || (a->score == b->score && a->pos < b->pos);
}

static void heap_push(BpeCand *h, int *n, BpeCand c) {
    int i = (*n)++;
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (!cand_before(&c, &h[p])) break;
        h[i] = h[p];
        i = p;
    }
    h[i] = c;
}

static BpeCand heap_pop(BpeCand *h, int *n) {
    BpeCand top = h[0], last = h[--(*n)];
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1;
        if (l >= *n) break;
        int m = (r < *n && cand_before(&h[r], &h[l])) ? r : l;
        if (!cand_before(&h[m], &last)) break;
        h[i] = h[m];
        i = m;
    }
    if (*n > 0) h[i] = last;
    return top;
}

static inline void bpe_push(Tokenizer *t, const unsigned char *chunk, BpeScratch *s, int *nh, int a, int b) {
    // symbols a and b are adjacent, so their concatenation is chunk[a .. a + len)
    int len = s->len[a] + s->len[b];
    if ((unsigned int)len > t->max_token_length) return;
    int id = vocab_lookup(t, chunk + a, len);
    if (id < 0) return;
    BpeCand c = { t->merge_scores[id], a, len, id };
    heap_push(s->heap, nh, c);
}

static void bpe_chunk(Tokenizer *t, const unsigned char *chunk, int n, BpeScratch *s, int *tokens, int *n_tokens) {
    if (n == 1) { tokens[(*n_tokens)++] = t->byte_token[chunk[0]]; return; }
    for (int i = 0; i < n; i++) {
        s->len[i] = 1;
        s->tok[i] = t->byte_token[chunk[i]];
        s->prev[i] = i - 1;
    }
    int nh = 0;
    for (int i = 0; i + 1 < n; i++) bpe_push(t, chunk, s, &nh, i, i + 1);
    while (nh > 0) {
        BpeCand c = heap_pop(s->heap, &nh);
        int i = c.pos;
        if (s->len[i] == 0) continue; // symbol merged into its left neighbour since
        int j = i + s->len[i];
        if (j >= n || s->len[i] + s->len[j] != c.len) continue; // stale: the pair changed
        s->tok[i] = c.id;
        s->len[i] = c.len;
        s->len[j] = 0;
        int k = i + c.len;
        if (k < n) s->prev[k] = i;
        if (i > 0) bpe_push(t, chunk, s, &nh, s->prev[i], i);
        if (k < n) bpe_push(t, chunk, s, &nh, i, k);
    }
    for (int i = 0; i < n; i += s->len[i]) tokens[(*n_tokens)++] = s->tok[i];
}

static void encode_segment(Tokenizer *t, const unsigned char *p, const unsigned char *end, BpeScratch *s, int *tokens, int *n_tokens) {
    // text between added tokens: NFC, pre-tokenize, then BPE each chunk
    unsigned char *norm = NULL;
    const unsigned char *q = nfc_check(p, end);
    if (q < end) {
        int n;
        norm = nfc_normalize(p, q, end, &n);
        p = norm;
        end = norm + n;
        bpe_reserve(s, n);
    }
    while (p < end) {
        q = pretok_next(p, end);
        bpe_chunk(t, p, (int)(q - p), s, tokens, n_tokens);
        p = q;
    }
    free(norm);
}

static int match_added(Tokenizer *t, const unsigned char *p, const unsigned char *end, int *len) {
    // longest added token starting at p (t->added is sorted longest first), or -1
    for (int k = 0; k < t->n_added; k++) {
        int id = t->added[k], n = t->vocab_len[id];
        if (n <= end - p && !memcmp(p, t->vocab[id], n)) { *len = n; return id; }
    }
    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array of
    // ENCODE_MAX_TOKENS(strlen(text)) ints
    const unsigned char *s = (const unsigned char *)text, *end = s + strlen(text);
    BpeScratch sc = {0};
    bpe_reserve(&sc, (int)(end - s));

    *n_tokens = 0;
    const unsigned char *seg = s;
    for (const unsigned char *p = s; p < end;) {
        int len, id = t->added_first[*p] ? match_added(t, p, end, &len) : -1;
        if (id < 0) { p++; continue; }
        encode_segment(t, seg, p, &sc, tokens, n_tokens);
        tokens[(*n_tokens)++] = id;
        p += len;
        seg = p;
    }
    encode_segment(t, seg, end, &sc, tokens, n_tokens);

    free(sc.len);
    free(sc.heap);
}

