#!/usr/bin/env python3
"""summary of s9c_bench JSONs: one-client / 4-client decode, prefill tok/s of the >= 700-token one-client prompts
(the first 3: think on, greedy: lcb1, lcb2, code; and the median of all such), long-run turns. usage: summ.py J..."""
import json, statistics, sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"== {path.split('/')[-1]} (peak VmHWM {d['peak'].get('VmHWM')} GiB, load {d.get('load_s')} s, {d['time']})")
    res = d["results"]
    if "short" in res:
        for m, v in res["short"].items():
            fc = " / ".join(f"{x['tok_s']:.1f}" for x in v["four_clients"])
            print(f"  {m}: 1 client {v['one_client_mean_tok_s']:.2f}, 4 clients greedy/seeded {fc}")
        one = [r for r in d["requests"] if "clients" not in r["key"] and r.get("prompt_n", 0) >= 700 and r.get("prompt_ms")]
        pp = [r["prompt_n"] / r["prompt_ms"] * 1000 for r in one]
        first3 = pp[:3]
        print(f"  prefill >=700-token prompts: first 3 (think on, greedy) {min(first3):.0f}-{max(first3):.0f}; all {len(pp)}: "
              f"median {statistics.median(pp):.0f}, range {min(pp):.0f}-{max(pp):.0f}")
    if "long" in res:
        L = res["long"]
        print(f"  cold prefill {L['prompt_n']} tokens {L['cold_prefill_ms'] / 1000:.0f} s ({L['prompt_n'] / L['cold_prefill_ms'] * 1000:.1f} tok/s)")
        for t in L["turns"]:
            print(f"  turn {t['turn']}: wall {t['wall_s']:.0f} s, prefill {t['prompt_n']} tokens {t['prompt_ms'] / 1000:.1f} s "
                  f"(cache_n {t['cache_n']}), decode {t['tok_s']:.2f} tok/s")
