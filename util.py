#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py

Runs:
- Base pipeline (pipeline.py): preprocess + features + sentiment + rule policy
- HF ensemble (pipeline_huggingface.py): zero-shot + few-shot + (optional ft) + ensemble
- FAST CLASSIFIER (new): TF-IDF + LinearSVC trained on rule-based weak labels
    -> Saves predictions, precision/recall/F1, confusion matrix (CSV/PNG), report.json

Usage examples:
  # Run all three and save into ./outputs
  python util.py --input reviews.csv --outdir outputs

  # Base only (skip HF + fast)
  python util.py --input reviews.csv --outdir outputs --run_hf false --run_fast_clf false

  # HF only (skip base + fast), with lite models & smaller batch:
  python util.py --input reviews.csv --outdir outputs --run_base false --run_fast_clf false \
      --lite true --batch_size 8 --max_rows 500

  # FAST only (skip base + HF)
  python util.py --input reviews.csv --outdir outputs --run_base false --run_hf false --run_fast_clf true
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np

# Import your modules (same folder)
import pipeline as base
import pipeline_huggingface as hf

# Fast ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_confusion_matrix_png(cm, labels, out_png: Path, title: str):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)

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
    batch_size: int = 8,
    lite: bool = False,
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
        batch_size=batch_size,
        lite=lite,
    )
    print(f"✅ HF pipeline saved: {res['out_csv']}")
    return str(res["out_csv"])

def run_fast_classifier(
    input_path: str,
    outdir: str,
    retrain: bool = False,
    min_df: int = 2,
    max_df: float = 0.95,
) -> str:
    """
    Ultra-fast classical ML model development:
      - uses base.preprocess + base.feature_engineering
      - uses base.apply_llm_sentiment(provider='none') for instant sentiment
      - uses base.assign_policy_category for weak labels
      - trains TF-IDF + LinearSVC to predict policy categories (incl. relevancy via "irrelevant")
      - saves metrics (if policy_gt exists), confusion matrix CSV/PNG, and report.json
    """
    outdir_p = Path(outdir) / "fast"
    outdir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = base.preprocess(df)
    df = base.feature_engineering(df)
    df = base.apply_llm_sentiment(df, provider="none")
    df = base.assign_policy_category(df, require_english=False)

    model_dir = outdir_p / "model_joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    vpath = model_dir / "vectorizer.joblib"
    mpath = model_dir / "classifier.joblib"

    if retrain:
        for p in [vpath, mpath]:
            if p.exists(): p.unlink()

    if vpath.exists() and mpath.exists():
        vec = joblib.load(vpath); clf = joblib.load(mpath); trained = False
    else:
        y = df["policy_category"].astype(str).fillna("clean").values
        X_texts = df["text"].fillna("").astype(str).values
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df)
        X = vec.fit_transform(X_texts)
        clf = LinearSVC()
        clf.fit(X, y)
        joblib.dump(vec, vpath); joblib.dump(clf, mpath)
        trained = True

    # Predict whole dataset
    X_all = vec.transform(df["text"].fillna("").astype(str).values)
    df["policy_fast_ml"] = clf.predict(X_all)

    # Simple ensemble: prefer ML if rule == spam_or_lowinfo (rules may over-flag)
    def choose(rule, ml):
        if rule == "spam_or_lowinfo": return ml
        return ml if ml != "clean" else rule
    df["policy_final_fast"] = [choose(r, m) for r, m in zip(df["policy_category"], df["policy_fast_ml"])]

    # Metrics
    report = {
        "dataset_rows": int(len(df)),
        "model_trained": bool(trained),
        "classes_seen": sorted(list(set(df["policy_category"].astype(str)))),
    }
    if "policy_gt" in df.columns:
        y_true = df["policy_gt"].astype(str).values
        y_pred = df["policy_final_fast"].astype(str).values
        labels = sorted(list(set(y_true) | set(y_pred)))
        clf_rep = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
        report["policy_metrics"] = clf_rep
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(outdir_p / "confusion_matrix_policy.csv")
        _save_confusion_matrix_png(cm, labels, outdir_p / "confusion_matrix_policy.png", "Policy Confusion Matrix")
    if "sentiment_gt" in df.columns:
        ys_true = df["sentiment_gt"].astype(str).values
        ys_pred = df["sentiment_label"].astype(str).values
        labs = sorted(list(set(ys_true) | set(ys_pred)))
        srep = classification_report(ys_true, ys_pred, labels=labs, zero_division=0, output_dict=True)
        report["sentiment_metrics"] = srep

    (outdir_p / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_csv = outdir_p / "fast_reviews.csv"
    df.to_csv(out_csv, index=False)

    print(f"✅ FAST classifier saved: {out_csv}")
    if "policy_gt" in df.columns:
        print("✅ FAST metrics saved:", outdir_p / "report.json")
        print("✅ Confusion matrix CSV/PNG saved in:", outdir_p)
    print(df[["text","sentiment_label","sentiment_score","policy_category","policy_fast_ml","policy_final_fast"]].head(8).to_string(index=False))
    return str(out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="reviews.csv", help="Input CSV path")
    ap.add_argument("--outdir", type=str, default="outputs", help="Output directory")

    # Which pipelines to run
    ap.add_argument("--run_base", type=str, default="true", help="true/false")
    ap.add_argument("--run_hf", type=str, default="true", help="true/false")
    ap.add_argument("--run_fast_clf", type=str, default="true", help="true/false")

    # Base pipeline options
    ap.add_argument("--require_english", action="store_true")
    ap.add_argument("--sentiment_provider", type=str, default="openai", choices=["openai","hf","none"])
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--hf_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--rate_limit_sleep", type=float, default=0.0)

    # HF pipeline options
    ap.add_argument("--policy_zs_model", type=str, default="facebook/bart-large-mnli")
    ap.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--finetuned_policy_ckpt", type=str, default="")
    ap.add_argument("--sentiment_model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lite", action="store_true")

    # FAST classifier options
    ap.add_argument("--retrain_fast", action="store_true")
    ap.add_argument("--fast_min_df", type=int, default=2)
    ap.add_argument("--fast_max_df", type=float, default=0.95)

    args = ap.parse_args()
    def _to_bool(s: str) -> bool: return str(s).strip().lower() in {"1","true","t","yes","y"}

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
            batch_size=args.batch_size,
            lite=args.lite,
        )

    if _to_bool(args.run_fast_clf):
        run_fast_classifier(
            input_path=args.input,
            outdir=args.outdir,
            retrain=args.retrain_fast,
            min_df=args.fast_min_df,
            max_df=args.fast_max_df,
        )

    print("\n🎉 Done.")

if __name__ == "__main__":
    main()
