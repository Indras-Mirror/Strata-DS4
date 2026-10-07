#!/bin/bash
# CPU smoke test of ds4_generate --verify on the mini fixtures (no GPU: CUDA_VISIBLE_DEVICES=).
# Every verify variant must print byte-identical tokens to plain greedy AND dump byte-identical logits rows behind
# each token (DS4_DUMP_TOKLOGITS) - on the CPU multi-token passes are bit-identical to one-token decoding.  Greedy
# tokens alone missed a deliberately skipped position on this fixture.
cd "$(dirname "$0")/../../.."
D=bench/ds4-2026-10-06/dense; O=bench/ds4-2026-10-07/verify-cpu
R="nice -n 10 systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 env CUDA_VISIBLE_DEVICES="
fail=0
# gen <out> args...: tokens -> <out>, logits rows -> <out>.lg, stderr -> <out>.err
gen() { local out=$1; shift
  $R DS4_DUMP_TOKLOGITS=$out.lg DS4_DUMP_DRAFTLOGITS=$out.dr ./build-ds4-gpu/ds4_generate -m $D/mini-ds4-tame.gguf --backend cpu --experts cpu \
     --arena-gib 1 "$@" > $out 2> $out.err; }
# chk <name> <ref tokens> <ref logits> <out>: out's tokens == ref tokens, out's logits == the ref's first rows
chk() {
  local n=$(wc -l < "$4") row=$(( $(stat -c %s "$3") / $(wc -l < "$PLAIN0") )) r=""
  [ -s "$2" ] && cmp -s "$2" "$4" || r="$r DIFF-tokens"
  [ "$(stat -c %s "$4.lg")" = $(( n * row )) ] && cmp -s -n $(( n * row )) "$3" "$4.lg" || r="$r DIFF-logits"
  # MTP draft logits vs plain --mtp's, wherever both drafted the same token index (needs the .mtp reference)
  [ -n "$5" ] && { python3 tools/ds4/cmp_draft_logits.py "$5.dr" "$4.dr" $(( row / 4 )) $([ $n = 1 ] && echo --allow-empty) > "$4.drcmp" || r="$r DIFF-drafts"; }
  [ -z "$r" ] && r=ok || fail=$((fail+1))
  printf "%-34s %-24s %3s tok | %s | %s\n" "$1" "$r" "$n" "$(grep -h '^verify : [0-9]' "$4.err" | sed 's/ tok\/s.*//')" \
    "$(grep -h '^MTP:' "$4.err" | cut -d'(' -f1)"
}
P=( "11,48,21,58,31,4,41,14" "3,7,9" "60,2,33,17,45,8,29,51,12,40,6,19" )
PLAIN0=$O/p0.plain
for i in 0 1 2; do
  ids=${P[$i]}; ref=$O/p$i.plain
  gen $ref --ids $ids -n 40
  gen $ref.mtp --ids $ids -n 40 --mtp $D/mini-ds4-mtp.gguf      # plain greedy + MTP drafts at every position
  chk "p$i plain --mtp" $ref $ref.lg $ref.mtp
  gen $O/p$i.v --ids $ids -n 40 --mtp $D/mini-ds4-mtp.gguf --verify
  chk "p$i verify (MTP drafts)" $ref $ref.lg $O/p$i.v $ref.mtp
  for c in 0 3 2; do
    DS4_VERIFY_ORACLE=$ref DS4_VERIFY_CORRUPT=$c gen $O/p$i.o$c --ids $ids -n 40 --mtp $D/mini-ds4-mtp.gguf --verify
    chk "p$i oracle corrupt=$c" $ref $ref.lg $O/p$i.o$c $ref.mtp
  done
done
# n_predict cuts (an accepted pass emits 2: the limit must cut between them), oracle = 100% accept
for n in 1 2 3 7 8; do
  head -n $n $PLAIN0 > $O/n$n.ref
  DS4_VERIFY_ORACLE=$PLAIN0 gen $O/n$n.v --ids ${P[0]} -n $n --mtp $D/mini-ds4-mtp.gguf --verify
  chk "p0 -n $n oracle" $O/n$n.ref $PLAIN0.lg $O/n$n.v $PLAIN0.mtp
done
# --stop at the k-th token's id (cuts at its first occurrence), both parities
for k in 5 6 9; do
  st=$(sed -n "${k}p" $PLAIN0); first=$(grep -n -m1 -x "$st" $PLAIN0 | cut -d: -f1)
  gen $O/s$k.ref --ids ${P[0]} -n 40 --stop $st
  for c in 0 3; do
    DS4_VERIFY_ORACLE=$PLAIN0 DS4_VERIFY_CORRUPT=$c gen $O/s$k.v$c --ids ${P[0]} -n 40 --stop $st --mtp $D/mini-ds4-mtp.gguf --verify
    chk "p0 --stop $st (first at $first) c=$c" $O/s$k.ref $PLAIN0.lg $O/s$k.v$c $PLAIN0.mtp
  done
done
echo "FAILURES: $fail"
