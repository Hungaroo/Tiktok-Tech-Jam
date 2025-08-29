#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py
One command to run:
- Your base pipeline (pipeline.py): preprocess + features + LLM sentiment + single-label policy
- The HF ensemble pipeline (pipeline_huggingface.py): zero-shot NLI + few-shot LLM + optional finetuned + ensemble

Usage examples:
  # Run both and save into ./outputs
  python util.py --input reviews.csv --outdir outputs

  # Base only (skip HF)
  python util.py --input reviews.csv --outdir outputs --run_hf false

  # HF only (skip base)
  python util.py --input reviews.csv --outdir outputs --run_base false \
      --policy_zs_model facebook/bart-large-mnli \
      --gen_model Qwen/Qwen2.5-7B-Instruct \
      --sentiment_model cardiffnlp/twitter-roberta-base-sentiment-latest
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Import your modules (same folder)
import pipeline as base
import pipeline_huggingface as hf


def run_base_pipeline(
    input_path: str,
    outdir: str,
    require_english: bool = False,
    sentiment_provider: str = "openai",   # "openai" | "hf" | "none"
    openai_model: str = "gpt-4o-mini",
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    rate_limit_sleep: float = 0.0,
) -> str:
    outdir_p = Path(outdir) / "base"
    outdir_p.mkdir(parents=True, exist_ok=True)
    out_csv = outdir_p / "reviews_processed_base.csv"

    df = pd.read_csv(input_path)
    df = base.preprocess(df)
    df = base.feature_engineering(df)
    df = base.apply_llm_sentiment(
        df,
        provider=sentiment_provider,
        openai_model=openai_model,
        hf_model_id=hf_model_id,
        rate_limit_sleep=rate_limit_sleep,
    )
    df = base.assign_policy_category(df, require_english=require_english)
    df.to_csv(out_csv, index=False)

    print(f"✅ Base pipeline saved: {out_csv}")
    print(df[["text","sentiment_label","sentiment_score","policy_category"]].head(5).to_string(index=False))
    return str(out_csv)


def run_hf_pipeline(
    input_path: str,
    outdir: str,
    policy_zs_model: str = "facebook/bart-large-mnli",
    gen_model: str = "Qwen/Qwen2.5-7B-Instruct",
    finetuned_policy_ckpt: str = "",
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_rows: int = 0,
) -> str:
    outdir_p = Path(outdir) / "hf"
    outdir_p.mkdir(parents=True, exist_ok=True)
    res = hf.run_hf_pipeline(
        input_path=input_path,
        outdir=str(outdir_p),
        policy_zs_model=policy_zs_model,
        gen_model=gen_model,
        finetuned_policy_ckpt=finetuned_policy_ckpt,
        sentiment_model=sentiment_model,
        max_rows=max_rows,
    )
    print(f"✅ HF pipeline saved: {res['out_csv']}")
    return str(res["out_csv"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="reviews.csv", help="Input CSV path")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    # Which pipelines to run
    ap.add_argument("--run_base", type=str, default="true", help="true/false")
    ap.add_argument("--run_hf", type=str, default="true", help="true/false")

    # Base pipeline options
    ap.add_argument("--require_english", action="store_true", help="Flag non-English reviews in base pipeline")
    ap.add_argument("--sentiment_provider", type=str, default="openai", choices=["openai","hf","none"])
    ap.add_argument("--openai_model", type=str, default="gpt-4.1", help="OpenAI model for sentiment (recommend gpt-4.1)")
    ap.add_argument("--hf_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model for sentiment if provider=hf")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.0)

    # HF pipeline options
    ap.add_argument("--policy_zs_model", type=str, default="facebook/bart-large-mnli")
    ap.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--finetuned_policy_ckpt", type=str, default="")
    ap.add_argument("--sentiment_model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--max_rows", type=int, default=0, help="limit rows for quick tests (0 = all)")

    args = ap.parse_args()

    # Normalize booleans
    def _to_bool(s: str) -> bool:
        return str(s).strip().lower() in {"1","true","t","yes","y"}

    if _to_bool(args.run_base):
        run_base_pipeline(
            input_path=args.input,
            outdir=args.outdir,
            require_english=args.require_english,
            sentiment_provider=args.sentiment_provider,
            openai_model=args.openai_model,
            hf_model_id=args.hf_model_id,
            rate_limit_sleep=args.rate_limit_sleep,
        )

    if _to_bool(args.run_hf):
        run_hf_pipeline(
            input_path=args.input,
            outdir=args.outdir,
            policy_zs_model=args.policy_zs_model,
            gen_model=args.gen_model,
            finetuned_policy_ckpt=args.finetuned_policy_ckpt,
            sentiment_model=args.sentiment_model,
            max_rows=args.max_rows,
        )

    print("\n🎉 Done.")

if __name__ == "__main__":
    main()
