#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reviews Pipeline: Preprocessing, LLM Sentiment (label + score), Feature Engineering, Single-Label Policy Category
- NO model training
- Profanity/hate filtering DISABLED
- Produces:
    - sentiment_label ∈ {negative, neutral, positive}
    - sentiment_score ∈ [-1, 1]
    - policy_category single label
Usage:
    python reviews_pipeline_llm_sentiment.py --input reviews.csv --output reviews_processed.csv \
        --sentiment_provider openai --openai_model gpt-4o-mini
"""

import os
import re
import json
import argparse
import time
from typing import Optional, Tuple

import pandas as pd
import numpy as np

# ---- Optional deps (graceful fallback) ----
_HAS_LANGDETECT = True
try:
    from langdetect import detect as _ld_detect  # pip install langdetect
except Exception:
    _HAS_LANGDETECT = False

_HAS_OPENAI = True
_OPENAI_MODE = None  # "client" (new) | "legacy" (old) | None
try:
    from openai import OpenAI  # pip install openai (>=1.0)
    _OPENAI_MODE = "client"
except Exception:
    try:
        import openai  # older SDK
        _OPENAI_MODE = "legacy"
    except Exception:
        _HAS_OPENAI = False
        _OPENAI_MODE = None

_HAS_TRANSFORMERS = True
try:
    from transformers import pipeline  # pip install transformers accelerate
except Exception:
    _HAS_TRANSFORMERS = False

# =========================
# Globals: regexes & config
# =========================
URL_RE     = re.compile(r"(https?://\S+|www\.[^\s]+)", re.IGNORECASE)
HTML_RE    = re.compile(r"<[^>]+>")
EMOJI_RE   = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
NONALNUM   = re.compile(r"[^a-z0-9\s']", re.IGNORECASE)
MULTISPC   = re.compile(r"\s+")
EMAIL_RE   = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE   = re.compile(r"(?:\+?\d[\s\-]?){7,}")
MAL_TLD_RE = re.compile(r"\.(ru|tk|cn|gq|ml|cf)(/|$)", re.IGNORECASE)  # simple suspicious TLDs

STOPWORDS = {
    "a","an","the","and","or","but","of","in","on","for","to","with","is","are","was","were",
    "be","am","i","you","he","she","it","we","they","this","that","these","those","at","as",
    "by","from","about","into","over","after","so","than","very","can","could","should","would",
    "have","has","had","do","does","did","not","no","yes","too","also","just","if","when","then",
    "there","here","out","up","down","more","most","less","least","much","many","my","your","his",
    "her","their","our","me","them","us"
}

# Policy dictionaries (core + extras)
AD_PROMO_WORDS    = {"promo","promotion","discount","voucher","coupon","sale","deal","code","offer"}
CTA_PHRASES       = {"visit ","order now","call now","book now","whatsapp","dm us","message us"}
NOT_VISIT_PHRASES = {"never been","havent been","haven't been","i heard","someone said","people say","didnt go","didn't go"}
IRRELEVANT_HINTS  = {"phone","phone model","iphone","android","laptop","computer","shipping","delivery","parcel","courier","headphones","charger"}

# Spam / low-info thresholds
MAX_CHAR_REPEAT     = 5      # e.g., "soooo" > 5 repeated chars
MAX_WORD_REPEAT_FR  = 0.60   # >60% of tokens are the same
MIN_USEFUL_WORDS    = 3      # too short / low info if <= this

# Self-promo heuristics
SELF_PRONOUNS = {"we","our","us"}
PROMO_TONE    = {"grand opening","best in town","limited time","special offer","use code","new branch"}

# Fallback lexicon for sentiment (if LLM unavailable)
_POS_LEX = {"good","great","amazing","excellent","tasty","fresh","friendly","love","nice","wonderful","perfect","best"}
_NEG_LEX = {"bad","terrible","awful","horrible","slow","cold","rude","disgusting","worst","dirty","hate","bland","overpriced","noisy"}

# =========================
# 1) Preprocess & Clean
# =========================
def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = URL_RE.sub(" ", s)        # strip urls
    s = HTML_RE.sub(" ", s)       # strip html
    s = EMOJI_RE.sub(" ", s)      # strip emoji
    s = s.lower()
    s = NONALNUM.sub(" ", s)      # keep letters, numbers, spaces, apostrophes
    s = MULTISPC.sub(" ", s).strip()
    return s

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure expected columns exist
    for col in ["business_name","author_name","text","rating_category"]:
        if col not in df.columns:
            df[col] = ""

    # Drop exact duplicate author+text pairs
    df = df.drop_duplicates(subset=["author_name", "text"], keep="first")

    # Fill/standardize core string cols
    for col in ["business_name","author_name","rating_category"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Standardize rating if present
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Clean review text
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text"].apply(clean_text)
    return df

# =========================
# 2) LLM Sentiment (label + score)
# =========================
SENTIMENT_ALLOWED = {"negative","neutral","positive"}
SENTIMENT_PROMPT = (
    'You are a sentiment rater for location reviews.\n'
    'Review: "{text}"\n\n'
    'Task:\n'
    'Return a compact JSON object with two fields:\n'
    '{"label": one of ["negative","neutral","positive"], "score": a float between -1 and 1}\n'
    'Do not include any other text.'
)

def _render_sentiment_prompt(text: str) -> str:
    safe = str(text).replace('"', '\\"').replace("\n", " ").strip()
    return SENTIMENT_PROMPT.replace("{text}", safe)

def _extract_json(s: str) -> Optional[dict]:
    if not s:
        return None
    # Try to find a JSON object anywhere in the string
    try:
        # quick path: exact JSON
        return json.loads(s)
    except Exception:
        pass
    # fallback: extract last {...}
    start = s.rfind("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = s[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None
    return None

def _parse_sentiment_json(s: str) -> Optional[Tuple[str, float]]:
    js = _extract_json(s)
    if not js or "label" not in js or "score" not in js:
        return None
    label = str(js["label"]).lower().strip()
    try:
        score = float(js["score"])
    except Exception:
        return None
    # clamp & validate
    score = max(-1.0, min(1.0, score))
    if label not in SENTIMENT_ALLOWED:
        return None
    return (label, score)

def _sentiment_openai(text: str, model: str="gpt-4o-mini") -> Optional[Tuple[str, float]]:
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return None
    prompt = _render_sentiment_prompt(text)
    try:
        if _OPENAI_MODE == "client":
            client = OpenAI()
            # Prefer chat completions (deterministic & short)
            try:
                resp = client.chat.completions.create(
                    model=model, temperature=0, max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
                out = resp.choices[0].message.content
                return _parse_sentiment_json(out)
            except Exception:
                # Fallback: responses API (newer)
                try:
                    resp = client.responses.create(
                        model=model, input=prompt, temperature=0, max_output_tokens=20
                    )
                    out = getattr(resp, "output_text", None) or ""
                    return _parse_sentiment_json(out)
                except Exception:
                    return None
        elif _OPENAI_MODE == "legacy":
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            resp = openai.ChatCompletion.create(
                model=model, temperature=0, max_tokens=20,
                messages=[{"role": "user", "content": prompt}],
            )
            out = resp["choices"][0]["message"]["content"]
            return _parse_sentiment_json(out)
    except Exception:
        return None
    return None

_hf_pipe = None
def _ensure_hf_pipe(model_id: str="Qwen/Qwen2.5-0.5B-Instruct"):
    global _hf_pipe
    if _hf_pipe is not None:
        return _hf_pipe
    if not _HAS_TRANSFORMERS:
        return None
    try:
        _hf_pipe = pipeline(
            "text-generation", model=model_id, tokenizer=model_id,
            device_map="auto", max_new_tokens=48, do_sample=False,
            temperature=0.0, repetition_penalty=1.05,
        )
        return _hf_pipe
    except Exception:
        return None

def _sentiment_hf(text: str, model_id: str="Qwen/Qwen2.5-0.5B-Instruct") -> Optional[Tuple[str, float]]:
    pipe = _ensure_hf_pipe(model_id)
    if pipe is None:
        return None
    try:
        out = pipe(_render_sentiment_prompt(text))[0]["generated_text"]
        return _parse_sentiment_json(out)
    except Exception:
        return None

def _sentiment_lexicon(text: str) -> Tuple[str, float]:
    """Fallback when LLM not available/invalid: tiny lexicon → (label, score)."""
    if not text:
        return ("neutral", 0.0)
    toks = set(re.findall(r"[a-z']+", text.lower()))
    pos = len(toks & _POS_LEX); neg = len(toks & _NEG_LEX)
    if pos == 0 and neg == 0:
        return ("neutral", 0.0)
    score = (pos - neg) / float(pos + neg)
    label = "positive" if score > 0.2 else "negative" if score < -0.2 else "neutral"
    return (label, max(-1.0, min(1.0, score)))

def apply_llm_sentiment(
    df: pd.DataFrame,
    provider: str = "openai",            # "openai" | "hf" | "none"
    openai_model: str = "gpt-4o-mini",
    hf_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
    rate_limit_sleep: float = 0.0
) -> pd.DataFrame:
    labels, scores = [], []
    for _, row in df.iterrows():
        txt = row.get("text", "") or ""
        res = None
        if provider == "openai":
            res = _sentiment_openai(txt, model=openai_model)
        elif provider == "hf":
            res = _sentiment_hf(txt, model_id=hf_model_id)
        # provider == "none" or fallthrough → res stays None

        if not res:
            res = _sentiment_lexicon(txt)

        lab, sco = res
        labels.append(lab)
        scores.append(float(sco))

        if rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)

    out = df.copy()
    out["sentiment_label"] = labels
    out["sentiment_score"] = scores
    return out

# =========================
# 3) Classic feature engineering (no TextBlob)
# =========================
def simple_keywords(text: str, top_k:int=5)->str:
    """Very light keywording: top-k non-stopword tokens by frequency."""
    if not text:
        return ""
    toks = [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in STOPWORDS and len(t) > 2]
    if not toks:
        return ""
    from collections import Counter
    c = Counter(toks)
    top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    return " ".join([w for w,_ in top])

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Length features
    df["char_len"]     = df["text_clean"].str.len()
    df["word_len"]     = df["text_clean"].str.split().apply(len)
    df["avg_word_len"] = (df["char_len"] / df["word_len"].replace(0, 1)).round(3)

    # Keywords
    df["keywords"] = df["text_clean"].apply(simple_keywords)

    # Optional metadata transforms if present
    if "timestamp" in df.columns:
        df["posting_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["post_hour"] = df["posting_time"].dt.hour
        df["post_weekday"] = df["posting_time"].dt.weekday
    if "user_id" in df.columns:
        df["user_review_count"] = df.groupby("user_id")["text"].transform("count")
    if {"gps_lat","gps_lon"} <= set(df.columns):
        df["gps_has_coords"] = (~df["gps_lat"].isna()) & (~df["gps_lon"].isna())

    # Optional language detection
    if _HAS_LANGDETECT:
        def _lang_or_unk(s):
            try:
                return _ld_detect(s) if s else "unk"
            except Exception:
                return "unk"
        df["lang"] = df["text"].apply(_lang_or_unk)
    else:
        df["lang"] = "unk"

    return df

# =========================
# 4) Single-label Policy Category (no profanity/hate)
# =========================
def is_advertisement(text: str) -> bool:
    t = text.lower()
    if "http" in t or "www." in t or bool(URL_RE.search(text)): 
        return True
    if any(w in t for w in AD_PROMO_WORDS): 
        return True
    if any(p in t for p in CTA_PHRASES): 
        return True
    return False

def is_irrelevant(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in IRRELEVANT_HINTS)

def is_rant_without_visit(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in NOT_VISIT_PHRASES)

def looks_like_spam(text: str) -> bool:
    t = text.lower()
    if len(t) == 0: 
        return True  # empty/near-empty = low value
    # excessive character repetition
    if re.search(r"(.)\1{%d,}" % MAX_CHAR_REPEAT, t): 
        return True
    # token-level checks
    toks = [w for w in re.findall(r"[a-z0-9']+", t) if w]
    if len(toks) <= MIN_USEFUL_WORDS: 
        return True
    # repeated word fraction
    if toks:
        from collections import Counter
        c = Counter(toks)
        top_freq = c.most_common(1)[0][1] / float(len(toks))
        if top_freq >= MAX_WORD_REPEAT_FR: 
            return True
    return False

def shares_personal_info(text: str) -> bool:
    return bool(EMAIL_RE.search(text) or PHONE_RE.search(text))

def non_english_when_en_required(text: str, require_en: bool=True) -> bool:
    if not require_en: 
        return False
    if not _HAS_LANGDETECT: 
        return False  # can't assess without detector
    try:
        lang = _ld_detect(text) if text else "unk"
        return lang not in {"en","unk"}
    except Exception:
        return False

def malicious_link(text: str) -> bool:
    t = text.lower()
    if not ("http" in t or "www." in t): 
        return False
    return bool(MAL_TLD_RE.search(t))

def self_promotion(text: str, business_name: str="") -> bool:
    t = text.lower()
    bn = (business_name or "").lower()
    cues = any(p in t for p in PROMO_TONE) or any(pr in t.split() for pr in SELF_PRONOUNS)
    bn_mentioned = bool(bn) and (bn in t)
    return cues and (bn_mentioned or "our" in t or "we" in t)

POLICY_PRIORITY = [
    ("advertisement",      lambda txt, row, req_en: is_advertisement(txt)),
    ("personal_info",      lambda txt, row, req_en: shares_personal_info(txt)),
    ("malicious_link",     lambda txt, row, req_en: malicious_link(txt)),
    ("self_promotion",     lambda txt, row, req_en: self_promotion(txt, row.get("business_name",""))),
    ("irrelevant",         lambda txt, row, req_en: is_irrelevant(txt)),
    ("rant_without_visit", lambda txt, row, req_en: is_rant_without_visit(txt)),
    ("non_english",        lambda txt, row, req_en: non_english_when_en_required(txt, require_en=req_en)),
    ("spam_or_lowinfo",    lambda txt, row, req_en: looks_like_spam(txt)),
]

def assign_policy_category(df: pd.DataFrame, require_english: bool=False) -> pd.DataFrame:
    cats = []
    for _, row in df.iterrows():
        txt = row.get("text", "") or ""
        label = None
        for name, fn in POLICY_PRIORITY:
            try:
                if fn(txt, row, require_english):
                    label = name
                    break
            except Exception:
                continue
        cats.append(label if label else "clean")
    out = df.copy()
    out["policy_category"] = cats
    return out

# =========================
# CLI Entrypoint
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="reviews.csv", help="Path to input CSV")
    parser.add_argument("--output", type=str, default="reviews_processed.csv", help="Path to save processed CSV")
    parser.add_argument("--require_english", action="store_true", help="Flag non-English reviews (requires langdetect)")
    parser.add_argument("--sentiment_provider", type=str, default="openai", choices=["openai","hf","none"],
                        help="Which provider to use for sentiment")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--hf_model_id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rate_limit_sleep", type=float, default=0.0, help="Sleep seconds between LLM calls")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # 1) Preprocess
    df = preprocess(df)

    # 2) Classic features (lengths, keywords, lang, etc.)
    df = feature_engineering(df)

    # 3) LLM sentiment (label + score)
    df = apply_llm_sentiment(
        df,
        provider=args.sentiment_provider,
        openai_model=args.openai_model,
        hf_model_id=args.hf_model_id,
        rate_limit_sleep=args.rate_limit_sleep
    )

    # 4) Single-label policy category
    df = assign_policy_category(df, require_english=args.require_english)

    # Save
    df.to_csv(args.output, index=False)

    print("✅ Saved:", args.output)
    print("Rows:", len(df))
    print("Sentiment label counts:\n", df["sentiment_label"].value_counts(dropna=False).to_string())
    print("\nPolicy category counts:\n", df["policy_category"].value_counts(dropna=False).to_string())
    print("\nPreview:")
    print(df[["text","sentiment_label","sentiment_score","policy_category"]].head(8))

if __name__ == "__main__":
    main()
