#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_huggingface.py
Ensemble / Multi-Task Inference (Hugging Face only, NO training)

Adds:
- --lite true      -> uses smaller/faster models
- --batch_size N   -> batch for zero-shot & sentiment calls (best-effort)
- --max_rows N     -> cap rows for quick dev

"""
from __future__ import annotations
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
except Exception as e:
    raise ImportError(
        "transformers is required for pipeline_huggingface.py. "
        "Install with: pip install transformers accelerate torch"
    ) from e

# =========================
# Labels / constants
# =========================
POLICY_LABELS = ["advertisement", "irrelevant", "rant_without_visit", "clean"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

POLICY_FEWSHOT_PROMPT = """You are a review moderation system.

Policies:
- advertisement: promotional content, marketing, links
- irrelevant: unrelated to the business/place
- rant_without_visit: complaints from someone who admits they never visited
- clean: valid review

Examples:
Review: "Best pizza! Visit www.pizzapromo.com for discounts!"
Label: advertisement

Review: "I love my new phone, but this place is too noisy."
Label: irrelevant

Review: "Never been here, but I heard it’s terrible."
Label: rant_without_visit

Review: "Amazing food and service!"
Label: clean

Now classify this review with EXACTLY one label from {labels}.
Review: "{text}"
Label:"""

SENTIMENT_JSON_PROMPT = """You are a sentiment rater for location reviews.
Return a compact JSON object: {"label": one of ["negative","neutral","positive"], "score": a float between -1 and 1}.
Do not include extra words.

Review: "{text}"
JSON:"""

# =========================
# Rule-based backstop
# =========================
import re
URL_RE     = re.compile(r"(https?://\S+|www\.[^\s]+)", re.IGNORECASE)
AD_PROMO   = {"promo","promotion","discount","voucher","coupon","sale","deal","code","offer"}
CTA_PHRASES= {"visit ","order now","call now","book now","whatsapp","dm us","message us"}
NOT_VISIT  = {"never been","havent been","haven't been","i heard","someone said","people say","didnt go","didn't go"}
IRRELEVANT = {"phone","phone model","iphone","android","laptop","computer","shipping","delivery","parcel","courier","headphones","charger"}

def rule_policy_label(text: str) -> str:
    t = (text or "").lower()
    if "http" in t or "www." in t or bool(URL_RE.search(t)): return "advertisement"
    if any(w in t for w in AD_PROMO): return "advertisement"
    if any(p in t for p in CTA_PHRASES): return "advertisement"
    if any(w in t for w in IRRELEVANT): return "irrelevant"
    if any(p in t for p in NOT_VISIT): return "rant_without_visit"
    return "clean"

# =========================
# Zero-shot policy via NLI (batched)
# =========================
def build_zs_policy_pipeline(model_id: str):
    return pipeline(
        "zero-shot-classification",
        model=model_id,
        device_map="auto",
        framework="pt",
    )

def predict_policy_zero_shot(zs_pipe, texts: List[str], batch_size:int=8) -> Tuple[List[str], List[Dict[str,float]]]:
    preds, scores = [], []
    if hasattr(zs_pipe, "batch"):
        iterator = (texts[i:i+batch_size] for i in range(0, len(texts), batch_size))
        for chunk in iterator:
            outs = zs_pipe(chunk, candidate_labels=POLICY_LABELS, multi_label=False)
            if isinstance(outs, dict): outs = [outs]
            for out in outs:
                label = out["labels"][0]
                d = {lab: float(sc) for lab, sc in zip(out["labels"], out["scores"])}
                d = {lab: d.get(lab, 0.0) for lab in POLICY_LABELS}
                preds.append(label); scores.append(d)
    else:
        for tx in texts:
            out = zs_pipe(tx, candidate_labels=POLICY_LABELS, multi_label=False)
            label = out["labels"][0]
            d = {lab: float(sc) for lab, sc in zip(out["labels"], out["scores"])}
            d = {lab: d.get(lab, 0.0) for lab in POLICY_LABELS}
            preds.append(label); scores.append(d)
    return preds, scores

# =========================
# Few-shot LLM for policy
# =========================
def build_generation_pipeline(model_id: str):
    return pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        device_map="auto",
        max_new_tokens=8,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        framework="pt",
    )

def _parse_policy_from_text(generated: str) -> Optional[str]:
    cand = (generated or "").strip().splitlines()[-1].strip().lower()
    cand = cand.replace("label:", "").strip().strip(".")
    mapping = {
        "ads":"advertisement", "ad":"advertisement", "promo":"advertisement",
        "promotional":"advertisement", "rant":"rant_without_visit",
        "rant without visit":"rant_without_visit", "no_visit_rant":"rant_without_visit",
    }
    cand = mapping.get(cand, cand)
    return cand if cand in POLICY_LABELS else None

def predict_policy_generation(gen_pipe, texts: List[str]) -> List[Optional[str]]:
    preds = []
    for tx in texts:
        prompt = POLICY_FEWSHOT_PROMPT.format(labels=POLICY_LABELS, text=str(tx).replace('"','\\"'))
        out = gen_pipe(prompt)[0]["generated_text"]
        preds.append(_parse_policy_from_text(out))
    return preds

# =========================
# Optional fine-tuned classifier
# =========================
def build_finetuned_policy_head(ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForSequenceClassification.from_pretrained(ckpt)
    return tok, mdl

def predict_policy_finetuned(tok, mdl, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
    preds, probs = [], []
    cls_pipe = pipeline(
        "text-classification",
        model=mdl,
        tokenizer=tok,
        device_map="auto",
        return_all_scores=True,
        framework="pt",
    )
    for tx in texts:
        out = cls_pipe(tx)[0]
        label_map = {d["label"].lower(): d["score"] for d in out}
        norm = {k.replace(" ", "_"): v for k, v in label_map.items()}
        scores = [float(norm.get(lab, 0.0)) for lab in POLICY_LABELS]
        preds.append(POLICY_LABELS[int(np.argmax(scores))])
        probs.append(scores)
    return preds, probs

# =========================
# Sentiment via HF (batched-ish best-effort)
# =========================
def build_sentiment_pipeline(model_id: str):
    return pipeline(
        "text-classification",
        model=model_id,
        device_map="auto",
        return_all_scores=True,
        framework="pt",
    )

def predict_sentiment(sent_pipe, texts: List[str], batch_size:int=16) -> Tuple[List[str], List[float]]:
    labels, scores = [], []
    if hasattr(sent_pipe, "batch"):
        iterator = (texts[i:i+batch_size] for i in range(0, len(texts), batch_size))
        for chunk in iterator:
            outs = sent_pipe(chunk)
            for out in outs:
                m = {d["label"].lower(): float(d["score"]) for d in out}
                if {"negative","neutral","positive"}.issubset(m.keys()):
                    lab = max(["negative","neutral","positive"], key=lambda k: m[k]); sc = m[lab]
                else:
                    key = max(m.keys(), key=lambda k: m[k])
                    lab = {"label_0":"negative","label_1":"neutral","label_2":"positive"}.get(key, key)
                    sc = m[key]
                labels.append(lab if lab in {"negative","neutral","positive"} else "neutral")
                pos = m.get("positive", m.get("label_2", 0.0))
                neg = m.get("negative", m.get("label_0", 0.0))
                scores.append(float(pos - neg))
    else:
        for tx in texts:
            out = sent_pipe(tx)[0]
            m = {d["label"].lower(): float(d["score"]) for d in out}
            if {"negative","neutral","positive"}.issubset(m.keys()):
                lab = max(["negative","neutral","positive"], key=lambda k: m[k]); sc = m[lab]
            else:
                key = max(m.keys(), key=lambda k: m[k])
                lab = {"label_0":"negative","label_1":"neutral","label_2":"positive"}.get(key, key)
                sc = m[key]
            labels.append(lab if lab in {"negative","neutral","positive"} else "neutral")
            pos = m.get("positive", m.get("label_2", 0.0))
            neg = m.get("negative", m.get("label_0", 0.0))
            scores.append(float(pos - neg))
    return labels, scores

# =========================
# Ensembling
# =========================
def ensemble_policy(rule_pred, zs_pred, gen_pred, ft_pred, weights: Dict[str, float] | None = None) -> str:
    if weights is None:
        weights = {"ft": 3.0, "zs": 2.0, "gen": 1.5, "rule": 1.0}
    votes = {lab: 0.0 for lab in POLICY_LABELS}
    if ft_pred in votes:   votes[ft_pred] += weights["ft"]
    if zs_pred in votes:   votes[zs_pred] += weights["zs"]
    if gen_pred in votes:  votes[gen_pred] += weights["gen"]
    if rule_pred in votes: votes[rule_pred] += weights["rule"]
    if all(v == 0.0 for v in votes.values()):
        return "clean"
    return max(votes.items(), key=lambda kv: kv[1])[0]

# =========================
# Public runner
# =========================
def run_hf_pipeline(
    input_path: str,
    outdir: str = "ensemble_outputs",
    policy_zs_model: str = "facebook/bart-large-mnli",
    gen_model: str = "Qwen/Qwen2.5-7B-Instruct",
    finetuned_policy_ckpt: str = "",
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    max_rows: int = 0,
    batch_size:int = 8,
    lite: bool = False,
) -> dict:
    outdir_p = Path(outdir); outdir_p.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    if max_rows and max_rows > 0:
        df = df.head(max_rows).copy()
    if "text" not in df.columns:
        raise ValueError("Input CSV must contain a 'text' column.")

    texts = df["text"].fillna("").astype(str).tolist()

    # Lite models (faster)
    if lite:
        policy_zs_model = "typeform/distilbert-base-uncased-mnli"
        gen_model       = "Qwen/Qwen2.5-0.5B-Instruct"
        sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"

    # Build components
    zs_pipe  = build_zs_policy_pipeline(policy_zs_model)
    gen_pipe = build_generation_pipeline(gen_model)
    sent_pipe= build_sentiment_pipeline(sentiment_model)

    tok = mdl = None
    if finetuned_policy_ckpt:
        tok, mdl = build_finetuned_policy_head(finetuned_policy_ckpt)

    # Inference
    rule_preds = [rule_policy_label(t) for t in texts]
    zs_preds, _   = predict_policy_zero_shot(zs_pipe, texts, batch_size=batch_size)
    gen_preds     = predict_policy_generation(gen_pipe, texts)
    ft_preds      = []
    if tok and mdl:
        ft_preds, _ = predict_policy_finetuned(tok, mdl, texts)
    sent_labels, sent_scores = predict_sentiment(sent_pipe, texts, batch_size=max(8, batch_size))

    # Ensemble
    ensemble_preds = []
    for i in range(len(texts)):
        rp  = rule_preds[i]
        zp  = zs_preds[i] if i < len(zs_preds) else None
        gp  = gen_preds[i] if i < len(gen_preds) else None
        ftp = ft_preds[i] if i < len(ft_preds) else None
        ensemble_preds.append(ensemble_policy(rp, zp, gp, ftp))

    # Attach + optional eval skeleton
    df_out = df.copy()
    df_out["policy_rule"]      = rule_preds
    df_out["policy_zero_shot"] = zs_preds
    df_out["policy_fewshot"]   = gen_preds
    if ft_preds:
        df_out["policy_finetuned"] = ft_preds
    df_out["policy_ensemble"]  = ensemble_preds
    df_out["sentiment_label_hf"] = sent_labels
    df_out["sentiment_score_hf"] = sent_scores

    report = {}
    if "policy_gt" in df_out.columns:
        def acc(col): return float((df_out[col] == df_out["policy_gt"]).mean())
        report["policy_accuracy"] = {
            "rule": acc("policy_rule"),
            "zero_shot": acc("policy_zero_shot"),
            "fewshot": acc("policy_fewshot"),
            **({"finetuned": acc("policy_finetuned")} if "policy_finetuned" in df_out.columns else {}),
            "ensemble": acc("policy_ensemble"),
        }
    if "sentiment_gt" in df_out.columns:
        report["sentiment_accuracy"] = float((df_out["sentiment_label_hf"] == df_out["sentiment_gt"]).mean())

    out_csv = outdir_p / "reviews_with_hf_ensemble.csv"
    df_out.to_csv(out_csv, index=False)
    out_json = outdir_p / "report.json"
    with open(out_json, "w") as f:
        json.dump({"metrics": report, "rows": int(len(df_out))}, f, indent=2)

    return {"df": df_out, "out_csv": str(out_csv), "out_json": str(out_json)}

# =========================
# CLI
# =========================
def _cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="reviews.csv")
    ap.add_argument("--outdir", type=str, default="ensemble_outputs")
    ap.add_argument("--policy_zs_model", type=str, default="facebook/bart-large-mnli")
    ap.add_argument("--gen_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--finetuned_policy_ckpt", type=str, default="")
    ap.add_argument("--sentiment_model", type=str, default="cardiffnlp/twitter-roberta-base-sentiment-latest")
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lite", action="store_true")
    args = ap.parse_args()

    res = run_hf_pipeline(
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

    print("✅ Saved:")
    print(" -", res["out_csv"])
    print(" -", res["out_json"])
    print("\nPreview:")
    dfp = res["df"][["text","policy_rule","policy_zero_shot","policy_fewshot","policy_ensemble","sentiment_label_hf","sentiment_score_hf"]]
    print(dfp.head(8).to_string(index=False))

if __name__ == "__main__":
    _cli()
