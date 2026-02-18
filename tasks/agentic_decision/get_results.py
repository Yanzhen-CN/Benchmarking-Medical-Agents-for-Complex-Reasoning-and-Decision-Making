import json
from pathlib import Path
from collections import defaultdict

def main(path: str = "results.json"):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # ---------- macro over patients ----------
    overall_acc_list = []
    patient_ids = []

    # ---------- micro over questions (via by_type.n * by_type.acc) ----------
    total_score = 0.0
    total_n = 0

    by_type_score = defaultdict(float)
    by_type_n = defaultdict(int)

    # ---------- mem0 config summary (optional) ----------
    mem0_configs = []

    for _, r in data.items():
        pid = r.get("patient_id") or r.get("patient") or r.get("pid")
        if pid:
            patient_ids.append(pid)

        if "overall_acc" in r and r["overall_acc"] is not None:
            overall_acc_list.append(float(r["overall_acc"]))

        bt = r.get("by_type") or {}
        for t, info in bt.items():
            if t in ("T3-M", "T3-P", "T3-A"):
                t = "T3-A"
            if not isinstance(info, dict):
                continue
            n = int(info.get("n", 0) or 0)
            acc = float(info.get("acc", 0.0) or 0.0)

            total_score += n * acc
            total_n += n

            by_type_score[t] += n * acc
            by_type_n[t] += n

        if isinstance(r.get("mem0"), dict):
            mem0_configs.append(r["mem0"])

    macro = sum(overall_acc_list) / len(overall_acc_list) if overall_acc_list else 0.0
    micro = total_score / total_n if total_n > 0 else 0.0

    print(f"#patients = {len(overall_acc_list)}")
    print(f"MACRO overall_acc (mean over patients): {macro:.6f}")
    print(f"MICRO overall_acc (weighted by n via by_type): {micro:.6f}")
    print(f"Total questions (sum over by_type.n): {total_n}")

    print("\nPer-type MICRO acc (weighted by n):")
    for t in sorted(by_type_n.keys()):
        acc_t = by_type_score[t] / by_type_n[t] if by_type_n[t] > 0 else 0.0
        print(f"  {t:4s}  n={by_type_n[t]:5d}  acc={acc_t:.6f}")

    # ---- mem0 config summary ----
    if mem0_configs:
        # normalize to tuples for comparison
        def freeze(d):
            return tuple(sorted((k, json.dumps(v, ensure_ascii=False, sort_keys=True)) for k, v in d.items()))
        uniq = {}
        for d in mem0_configs:
            uniq.setdefault(freeze(d), d)

        print("\nmem0 config:")
        if len(uniq) == 1:
            only = next(iter(uniq.values()))
            print("  (all same)")
            for k in sorted(only.keys()):
                print(f"  {k}: {only[k]}")
        else:
            print(f"  (mixed configs: {len(uniq)})")
            for i, d in enumerate(uniq.values(), 1):
                print(f"  -- config #{i} --")
                for k in sorted(d.keys()):
                    print(f"  {k}: {d[k]}")

if __name__ == "__main__":
    
    for log in (Path("agents/llm_agent/agentic_decision/results").glob("*/*.log")):
        print(f"\n\n{log.name}")
        main(log)
        
    for log in (Path("agents/llm_agent/agentic_decision/results").glob("*.log")):
        print(f"\n\n{log.name}")
        main(log)
        
    log = Path("/data/xzh/Benchmarking-Medical-Agents-for-Complex-Reasoning-and-Decision-Making/agents/rag_agent/agentic_decision/results/rag_eval_event_0.0_qwen-turbo_16_200_include_cutoff.json")
    print(f"\n\n{log.name}")
    main(log)
    
    log = Path("agents/rag_agent/agentic_decision/results/rag_eval_note_0.0_qwen-turbo_16_200_include_cutoff/rag_eval_note_0.0_qwen-turbo_16_200_include_cutoff.json")
    print(f"\n\n{log.name}")
    main(log)
    
    for log in (Path("agents/mem0_agent/agentic_decision/results").glob("mem0_*.json")):
        print(f"\n\n{log.name}")
        main(log)
    
    
