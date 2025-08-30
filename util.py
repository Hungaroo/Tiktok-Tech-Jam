#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
util.py

Runs:
- Base pipeline (pipeline.py): preprocess + features + sentiment + rule policy
- HF ensemble (pipeline_huggingface.py): zero-shot + few-shot + (optional ft) + ensemble
- FAST CLASSIFIER: TF-IDF + LinearSVC trained on rule-based weak labels
    -> Saves predictions to CSV
    -> If ground truth exists, saves ONE PNG confusion matrix based on the TRAINED MODEL ONLY
       (policy_fast_ml vs ground truth) with overall Accuracy, Macro-Precision/Recall/F1 in the title.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Import your modules (same folder)
import pipeline as base
import classifier as hf

# Fast ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Plotting (non-interactive backend)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _print_header(s: str):
    bar = "─" * max(8, len(s))
    print(f"\n{bar}\n{s}\n{bar}")

def _save_confusion_overall_png(y_true, y_pred, labels, out_png: Path, normalize: bool = True, hide_labels: bool = False):
    """
    Save ONE PNG with the confusion matrix (counts or row-normalized) and overall metrics in the title.
    Overall metrics only: Accuracy, Macro-Precision, Macro-Recall, Macro-F1.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    macro_p = rep["macro avg"]["precision"]
    macro_r = rep["macro avg"]["recall"]
    macro_f1 = rep["macro avg"]["f1-score"]

    disp = cm.copy()
    if normalize:
        row_sums = disp.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            disp = np.divide(disp, row_sums, where=row_sums != 0)
        disp[np.isnan(disp)] = 0.0

    # Figure sizing
    n = max(6, min(len(labels), 20))
    fig_w = min(14, max(6, n * 0.6))
    fig_h = min(12, max(5, n * 0.6))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(disp)
    if not hide_labels:
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    else:
        ax.set_xticks([]); ax.set_yticks([])

    ax.set_xlabel("Predicted (trained model)")
    ax.set_ylabel("True")
    title = f"Confusion Matrix (Trained TF-IDF+LinearSVC) | Acc={acc:.4f}  Macro P/R/F1=({macro_p:.3f}/{macro_r:.3f}/{macro_f1:.3f})"
    if normalize: title += " [normalized]"
    ax.set_title(title)

    # Only overall metrics; no per-class text files.
    if len(labels) <= 20:  # keep readable
        for i in range(disp.shape[0]):
            for j in range(disp.shape[1]):
                txt = f"{disp[i, j]:.2f}" if normalize else f"{int(disp[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"\n✅ Saved confusion matrix (trained model): {out_png}")
    print(f"   Accuracy={acc:.6f}  Macro-Precision={macro_p:.6f}  Macro-Recall={macro_r:.6f}  Macro-F1={macro_f1:.6f}")


# ----------------------------
# Base pipeline (no training)
# ----------------------------
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


# ----------------------------
# HF ensemble (inference only)
# ----------------------------
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


# ----------------------------
# FAST classifier (trains TF-IDF + LinearSVC)
# ----------------------------
def run_fast_classifier(
    input_path: str,
    outdir: str,
    retrain: bool = False,
    min_df: int = 2,
    max_df: float = 0.95,
    gt_col: str = "policy_gt",
    hide_labels: bool = True,   # hide class names on the matrix image if you want "overall only" visual
) -> str:
    """
    Classical ML:
      - preprocess + features
      - sentiment (provider='none')
      - rule-based weak labels (for training)
      - TF-IDF + LinearSVC training
      - Evaluate TRAINED MODEL ONLY (policy_fast_ml vs ground truth) -> ONE PNG with overall metrics
    """
    outdir_p = Path(outdir) / "fast"
    outdir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    df = base.preprocess(df)
    df = base.feature_engineering(df)
    df = base.apply_llm_sentiment(df, provider="none")
    df = base.assign_policy_category(df, require_english=False)  # weak labels for training

    model_dir = outdir_p / "model_joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    vpath = model_dir / "vectorizer.joblib"
    mpath = model_dir / "classifier.joblib"

    if retrain:
        for p in [vpath, mpath]:
            if p.exists(): p.unlink()

    if vpath.exists() and mpath.exists():
        vec = joblib.load(vpath); clf = joblib.load(mpath)
    else:
        y_weak = df["policy_category"].astype(str).fillna("clean").values
        X_texts = df["text"].fillna("").astype(str).values
        vec = TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df)
        X = vec.fit_transform(X_texts)
        clf = LinearSVC()
        clf.fit(X, y_weak)
        joblib.dump(vec, vpath); joblib.dump(clf, mpath)

    # Predict over the entire dataset USING TRAINED MODEL
    X_all = vec.transform(df["text"].fillna("").astype(str).values)
    df["policy_fast_ml"] = clf.predict(X_all)  # trained model predictions

    # Save predictions CSV (for reference)
    out_csv = outdir_p / "fast_reviews.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✅ FAST classifier predictions saved: {out_csv}")

    # --- EVALUATE TRAINED MODEL ONLY ---
    # Resolve ground-truth column (case-insensitive)
    col_map = {c.strip().lower(): c for c in df.columns}
    wanted = gt_col.strip().lower()
    if wanted in col_map:
        gt_name = col_map[wanted]
        y_true = df[gt_name].astype(str).fillna("NA").values
        y_pred = df["policy_fast_ml"].astype(str).fillna("NA").values  # trained model only
        labels = sorted(list(set(y_true) | set(y_pred)))
        out_png = outdir_p / "confusion_matrix_trained.png"
        _save_confusion_overall_png(y_true, y_pred, labels, out_png, normalize=True, hide_labels=hide_labels)
    else:
        _print_header("Ground-truth column not found")
        print(f"Looked for column '{gt_col}'.")
        print("Available columns:", list(df.columns))
        print("\n➡ Fix options:")
        print(f"  • Rename your GT column to '{gt_col}', OR")
        print("  • Re-run with your column name, e.g.:")
        print("    python util.py --input reviews.csv --outdir outputs "
              "--run_base false --run_hf false --run_fast_clf true "
              "--gt_col label")

    # Small preview
    print(df[["text","policy_category","policy_fast_ml"]].head(8).to_string(index=False))
    return str(out_csv)


# ----------------------------
# CLI
# ----------------------------
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
    ap.add_argument("--gt_col", type=str, default="policy_gt", help="Ground-truth column name (default: policy_gt)")
    ap.add_argument("--show_labels", action="store_true", help="Show class names on the matrix axes")

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
            gt_col=args.gt_col,
            hide_labels=not args.show_labels,
        )

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
