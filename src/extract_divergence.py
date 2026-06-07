"""
Extract per-token base<->instruct divergence features.

For each instruction:
  1. Generate a greedy response with the INSTRUCT model (chat-formatted prompt).
  2. Teacher-force the identical full token sequence through BOTH base and instruct
     models under the SAME chat-formatted context (isolates the weight difference).
  3. At each response position compute the divergence target + base-side features.

Output: a per-token parquet/csv table with one row per response token.

Targets:
  kl         = KL(P_inst || P_base)               (primary)
  base_rank  = rank of emitted token under P_base  (URIAL's eta; 1 = unshifted)
  logprob_delta = log P_inst(tok) - log P_base(tok)  (EFT / proxy behavior delta)
Base-side features (the "predictable" structure to subtract out):
  pos        = response-position index (0-based)
  base_entropy, inst_entropy
  base_top1_prob       = max_v P_base
  base_tok_logprob     = log P_base(emitted token)
  agree                = 1 if argmax P_base == argmax P_inst (top-1 agreement)
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_pair(name_base, name_inst, dev_base="cuda:0", dev_inst="cuda:1"):
    tok = AutoTokenizer.from_pretrained(name_inst)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        name_base, torch_dtype=torch.float16).to(dev_base).eval()
    inst = AutoModelForCausalLM.from_pretrained(
        name_inst, torch_dtype=torch.float16).to(dev_inst).eval()
    assert base.config.vocab_size == inst.config.vocab_size, "vocab mismatch"
    return tok, base, inst, dev_base, dev_inst


@torch.no_grad()
def generate_responses(tok, inst, dev_inst, instructions, max_new=96, bs=16):
    """Batch greedy generation with the instruct model (left padding)."""
    tok.padding_side = "left"
    responses = []
    for i in range(0, len(instructions), bs):
        batch = instructions[i:i + bs]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": ins}],
            tokenize=False, add_generation_prompt=True) for ins in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(dev_inst)
        out = inst.generate(**enc, max_new_tokens=max_new, do_sample=False,
                            num_beams=1, pad_token_id=tok.pad_token_id)
        gen = out[:, enc.input_ids.shape[1]:]
        for g in gen:
            responses.append(tok.decode(g, skip_special_tokens=True))
    return responses


@torch.no_grad()
def per_token_divergence(tok, base, inst, dev_base, dev_inst,
                         instruction, response, meta):
    """Teacher-force prompt+response through both models; return list of row dicts."""
    # Identical chat-formatted context for BOTH models (isolate weights).
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(response, add_special_tokens=False).input_ids
    if len(r_ids) == 0:
        return []
    full = torch.tensor([p_ids + r_ids])
    n_p = len(p_ids)
    # positions in `full` whose NEXT token is a response token: [n_p-1 .. len-2]
    lo, hi = n_p - 1, full.shape[1] - 1            # predict r_ids[0..]
    li = inst(full.to(dev_inst)).logits[0, lo:hi].float()   # [R, V]
    lb = base(full.to(dev_base)).logits[0, lo:hi].float().to(dev_inst)
    logp_i = F.log_softmax(li, dim=-1)
    logp_b = F.log_softmax(lb, dim=-1)
    p_i = logp_i.exp()
    p_b = logp_b.exp()
    kl = (p_i * (logp_i - logp_b)).sum(-1)                  # KL(inst||base) [R]
    base_ent = -(p_b * logp_b).sum(-1)
    inst_ent = -(p_i * logp_i).sum(-1)
    base_top1 = p_b.max(-1).values
    inst_argmax = logp_i.argmax(-1)
    base_argmax = logp_b.argmax(-1)
    agree = (inst_argmax == base_argmax)
    tgt = torch.tensor(r_ids, device=dev_inst)             # emitted tokens [R]
    base_tok_logp = logp_b.gather(-1, tgt[:, None]).squeeze(-1)
    inst_tok_logp = logp_i.gather(-1, tgt[:, None]).squeeze(-1)
    # base-rank of emitted token (1 = top): count tokens with higher base logprob + 1
    base_rank = (logp_b > base_tok_logp[:, None]).sum(-1) + 1

    R = len(r_ids)
    rows = []
    for t in range(R):
        rows.append(dict(
            **meta,
            pos=t,
            token_id=int(r_ids[t]),
            token_str=tok.decode([r_ids[t]]),
            kl=float(kl[t]),
            base_rank=int(base_rank[t]),
            logprob_delta=float(inst_tok_logp[t] - base_tok_logp[t]),
            base_entropy=float(base_ent[t]),
            inst_entropy=float(inst_ent[t]),
            base_top1_prob=float(base_top1[t]),
            base_tok_logprob=float(base_tok_logp[t]),
            agree=int(agree[t]),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--inst", required=True)
    ap.add_argument("--dataset", required=True, choices=["dolly", "just-eval"])
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev_base", default="cuda:0")
    ap.add_argument("--dev_inst", default="cuda:1")
    args = ap.parse_args()

    t0 = time.time()
    if args.dataset == "dolly":
        ds = load_from_disk("datasets/dolly-15k/hf")["train"]
        ds = ds.filter(lambda x: x["context"].strip() == "")   # clean instructions
        idx = np.random.RandomState(SEED).permutation(len(ds))[:args.n]
        rows_meta = [dict(instruction=ds[int(i)]["instruction"],
                          category=ds[int(i)]["category"], example_id=int(i))
                     for i in idx]
    else:
        ds = load_from_disk("datasets/just-eval-instruct/hf")["test"]
        n = min(args.n, len(ds))
        rows_meta = [dict(instruction=ds[i]["instruction"],
                          category=ds[i]["category"],
                          task=";".join(ds[i]["task"]) if ds[i]["task"] else "",
                          topic=";".join(ds[i]["topic"]) if ds[i]["topic"] else "",
                          example_id=int(ds[i]["id"])) for i in range(n)]

    instructions = [m["instruction"] for m in rows_meta]
    print(f"[{args.dataset}] {len(instructions)} instructions")

    tok, base, inst, db, di = load_pair(args.base, args.inst,
                                        args.dev_base, args.dev_inst)
    print(f"loaded pair vocab={base.config.vocab_size} ({time.time()-t0:.0f}s)")

    responses = generate_responses(tok, inst, di, instructions, args.max_new)
    print(f"generated ({time.time()-t0:.0f}s)")

    all_rows = []
    for j, (ins, resp, meta) in enumerate(zip(instructions, responses, rows_meta)):
        m = {k: v for k, v in meta.items() if k != "instruction"}
        all_rows.extend(per_token_divergence(tok, base, inst, db, di, ins, resp, m))
        if (j + 1) % 200 == 0:
            print(f"  {j+1}/{len(instructions)} ({time.time()-t0:.0f}s) "
                  f"rows={len(all_rows)}")
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    print(f"[done] {len(df)} token rows -> {args.out} ({time.time()-t0:.0f}s)")
    # quick sanity
    print(f"  top-1 agreement={df.agree.mean():.3f}  mean KL={df.kl.mean():.3f}  "
          f"median KL={df.kl.median():.3f}")


if __name__ == "__main__":
    main()
