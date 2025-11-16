# filename: main.py
import os
import argparse
import asyncio
import json

from model.coldrag_qwen import ColdRAG_qwen

def parse_args():
    ap = argparse.ArgumentParser("Qwen/vLLM runner")
    ap.add_argument("--model", required=True, help="Model name")
    ap.add_argument("--dataset", required=True, help="Dataset name (folder under ./dataset)")
    ap.add_argument("--core", type=int, required=True, help="Core level (e.g., 15)")
    ap.add_argument("--mode", default="coldrag", choices=["coldrag", "hybrid"], help="QueryParam.mode")
    ap.add_argument("--k", type=int, default=20, help="Top-K for ranking")
    ap.add_argument("--cand_size", type=int, default=100, help="Candidate list size (fallback sampling)")
    ap.add_argument("--batch_size", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--index_limit", type=int, default=None, help="Limit #txt docs to index (debug)")
    ap.add_argument("--skip_index", action="store_true", help="Skip indexing step")
    ap.add_argument("--out", default="./outputs/preds.json", help="Output predictions JSON")
    return ap.parse_args()

async def main():
    args = parse_args()

    candidate_path = f"./dataset/{args.dataset}/processed/candidate_list_{args.cand_size}_{args.core}.json"
    if not os.path.exists(candidate_path):
        raise FileNotFoundError(f"Candidate list not found at {candidate_path}")
    with open(candidate_path, "r") as f:
        candidate_lists = json.load(f)


    coldrag = ColdRAG_qwen(
        dataset=args.dataset,
        core=args.core,
        candidate_list_size=args.cand_size,
        mode=args.mode,
    )
    await coldrag.initialize()
    # import pdb; pdb.set_trace()
    if not args.skip_index:
        await coldrag.run_indexing(limit=args.index_limit)

    if not os.path.exists(args.out):
        preds = await coldrag.run_coldrag(
            output_path=args.out,
            k=args.k,
            batch_size=args.batch_size,
        )
    else:
        print(f"Skipping run_coldrag, loading existing results from {args.out}")
        with open(args.out, 'r') as file:
            preds = json.load(file)

    r, n, m = coldrag.evaluate_coldrag(preds, k=args.k)
    output_path = args.out

    print(f"[EVAL] Recall@{args.k}={r:.4f}  NDCG@{args.k}={n:.4f}  MRR={m:.4f}")
    eval_metrics = {f"Recall@{args.k}":r, f"NDCG@{args.k}":n, f"MRR@{args.k}":m}

    eval_output_path = output_path.replace(".json", "_eval.json")
    eval_data = {
        "model": args.model,
        "dataset": args.dataset,
        "k": args.k,
        "metrics": eval_metrics
    }
    with open(eval_output_path, 'w') as file:
        json.dump(eval_data, file, indent=2)

    print(f"Evaluation results saved to {eval_output_path}")

if __name__ == "__main__":
    asyncio.run(main())
