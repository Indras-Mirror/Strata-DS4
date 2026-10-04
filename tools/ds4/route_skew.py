"""Hit rate a VRAM expert cache would get on DeepSeek-V4, from route_probe's dump. Static cache = the most-used
(layer, expert) pairs on the train half; scored on the held-out half."""
import numpy as np, sys, collections
L, E, MIB_PER_EXPERT = 43, 256, 6.75
rec = np.fromfile(sys.argv[1], dtype=np.uint16).reshape(-1, 2)
layer, expert = rec[:, 0].astype(int), rec[:, 1].astype(int)
K = 6
# token index per layer: records arrive layer by layer, K per token, tokens in order
tok = np.zeros(len(rec), dtype=np.int64)
for l in range(L):
    idx = np.nonzero(layer == l)[0]
    tok[idx] = np.arange(len(idx)) // K
train = (tok // 512) % 2 == 0
def counts(mask):
    c = np.zeros((L, E)); np.add.at(c, (layer[mask], expert[mask]), 1); return c
tr, te = counts(train), counts(~train)
print(f"tokens/layer {tok.max()+1}, lookups train {int(tr.sum())} test {int(te.sum())}")
order = np.argsort(-tr.ravel())
te_sorted_by_train = te.ravel()[order]
te_oracle = np.sort(te.ravel())[::-1]
tot = te.sum()
print(f"{'VRAM for experts':>18} {'slots':>6} {'% of experts':>12} {'hit (static, held-out)':>22} {'oracle':>7} {'uniform':>7}")
for gb in (4, 8, 12, 16):
    n = int(gb * 1024 / MIB_PER_EXPERT)
    print(f"{gb:>15} GB {n:>6} {100*n/(L*E):>11.1f}% {100*te_sorted_by_train[:n].sum()/tot:>21.1f}% "
          f"{100*te_oracle[:n].sum()/tot:>6.1f}% {100*n/(L*E):>6.1f}%")
for target in (0.5, 0.8, 0.9):
    n = int(np.searchsorted(np.cumsum(te_sorted_by_train) / tot, target)) + 1
    print(f"experts needed for {int(target*100)}% held-out hits: {n} ({100*n/(L*E):.1f}%, {n*MIB_PER_EXPERT/1024:.1f} GB)")
# per-layer concentration
p = (tr + te) / (tr + te).sum(1, keepdims=True)
ent = np.exp(-(np.where(p > 0, p * np.log(p), 0)).sum(1))
print("effective experts per layer (perplexity of usage, max 256):",
      " ".join(f"{int(x)}" for x in ent))
