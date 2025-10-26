"""
FastAPI backend for Ethoscore Article Framing Analyzer.

Exposes endpoints for:
- Single article via URL: extract content, analyze, return combined results
- Single article via text: validate input, analyze
- Batch analysis via list of URLs or CSV upload

This service wraps the existing analysis logic in `src.model_analyzer` and
article extraction in `src.article_processor`.
"""

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dataclasses import is_dataclass, asdict
import json
import numpy as np
import pandas as pd
try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

# Local imports from this repository
from src.model_analyzer import (
    create_analyzer,
    ArticleFramingAnalyzer,
    ModelInferenceError,
    ModelLoadingError,
)
from src.article_processor import (
    process_url_input,
    process_manual_text_input,
    ArticleProcessingError,
    URLValidationError,
    ArticleExtractionError,
)
from src.batch_processor import BatchProcessor, parse_csv_input


app = FastAPI(title="Ethoscore API", version="1.0.0")

# CORS: allow local dev UIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "*",  # safe for local dev; consider tightening in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global analyzer instance
analyzer: Optional[ArticleFramingAnalyzer] = None
df_framing: Optional[pd.DataFrame] = None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable Python types.

    Handles numpy scalars/arrays, torch tensors, dataclasses, sets/tuples, etc.
    """
    # Fast-path for primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Dataclasses
    if is_dataclass(obj):
        return _sanitize_for_json(asdict(obj))

    # Numpy
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()

    # Torch tensors
    if torch is not None:
        try:
            if isinstance(obj, torch.Tensor):  # type: ignore[attr-defined]
                if obj.ndim == 0:
                    return _sanitize_for_json(obj.item())
                return obj.detach().cpu().tolist()
        except Exception:
            pass

    # Collections
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(x) for x in obj]

    # Fallback to string representation if still unserializable
    try:
        json.dumps(obj)  # probe
        return obj
    except Exception:
        return str(obj)


@app.on_event("startup")
def startup_event() -> None:
    global analyzer
    global df_framing
    try:
        analyzer = create_analyzer()
    except ModelLoadingError as err:
        # Defer raising; health endpoint will report failure
        analyzer = None
        # log via print to avoid configuring logging here
        print(f"[startup] Model loading failed: {err}")
    # Lazy dataset load on first use
    df_framing = None


class AnalyzeUrlRequest(BaseModel):
    url: str


class AnalyzeTextRequest(BaseModel):
    title: str
    body: str


class BatchUrlsRequest(BaseModel):
    urls: List[str]

class ExploreTopicRequest(BaseModel):
    keyword: str
    limit: int | None = 1
    label: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": analyzer is not None,
        "models": (analyzer.get_model_info() if analyzer else {"is_initialized": False}),
        "dataset_loaded": df_framing is not None,
    }


def require_analyzer() -> ArticleFramingAnalyzer:
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return analyzer


@app.post("/analyze/url")
def analyze_from_url(payload: AnalyzeUrlRequest) -> Dict[str, Any]:
    a = require_analyzer()
    try:
        title, body, article_info = process_url_input(payload.url)
        results = a.analyze_article(title, body)
        # Merge details for convenience on the client
        merged = {
            **results,
            "title": results.get("title", title),
            "source": article_info.get("source"),
            "source_url": article_info.get("source_url"),
            "publish_date": article_info.get("publish_date"),
            "authors": article_info.get("authors", []),
            "body_preview": body[:400],
        }
        return _sanitize_for_json(merged)
    except (URLValidationError, ArticleExtractionError, ArticleProcessingError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/analyze/text")
def analyze_from_text(payload: AnalyzeTextRequest) -> Dict[str, Any]:
    a = require_analyzer()
    try:
        clean_title, clean_body, article_info = process_manual_text_input(payload.title, payload.body)
        results = a.analyze_article(clean_title, clean_body)
        merged = {
            **results,
            "title": results.get("title", clean_title),
            "source": None,
            "source_url": None,
            "publish_date": None,
            "authors": [],
            "body_preview": clean_body[:400],
        }
        return _sanitize_for_json(merged)
    except ArticleProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelInferenceError as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/batch/urls")
async def analyze_batch_urls(payload: BatchUrlsRequest) -> Dict[str, Any]:
    a = require_analyzer()
    processor = BatchProcessor(a)
    results = await processor.process_urls_batch(payload.urls)
    # Convert dataclasses to dicts and sanitize
    payload_out = {
        "results": [r.__dict__ for r in results],
        "summary": processor.get_summary_stats(),
    }
    return _sanitize_for_json(payload_out)


@app.post("/batch/csv")
async def analyze_batch_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    a = require_analyzer()
    try:
        content_bytes = await file.read()
        content = content_bytes.decode("utf-8", errors="ignore")
        records = parse_csv_input(content)
        urls = [rec["url"] for rec in records]
        processor = BatchProcessor(a)
        results = await processor.process_urls_batch(urls)
        payload_out = {
            "results": [r.__dict__ for r in results],
            "summary": processor.get_summary_stats(),
        }
        return _sanitize_for_json(payload_out)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {e}")


def _ensure_dataset_loaded() -> pd.DataFrame:
    global df_framing
    if df_framing is None:
        try:
            df_framing = pd.read_csv("Dataset-framing_annotations-Llama-3.3-70B-Instruct-Turbo.csv")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
    return df_framing


@app.post("/explore/topic")
def explore_topic(payload: ExploreTopicRequest) -> Dict[str, Any]:
    a = require_analyzer()
    df = _ensure_dataset_loaded()
    keyword = (payload.keyword or "").strip()
    if not keyword:
        raise HTTPException(status_code=422, detail="keyword is required")

    # Prefer concept match; fallback to contains; then title/body search
    subset = df[df["concept"].astype(str).str.lower() == keyword.lower()]
    if subset.empty:
        subset = df[df["concept"].astype(str).str.contains(keyword, case=False, na=False)]
    if subset.empty:
        mask = (
            df["title"].astype(str).str.contains(keyword, case=False, na=False)
            | df["body"].astype(str).str.contains(keyword, case=False, na=False)
        )
        subset = df[mask]
    if subset.empty:
        return {"results": []}

    limit = max(1, int(payload.limit or 1))
    requested_label = (payload.label or "").strip().lower()
    # Iterate through a shuffled subset to improve chances of filter match
    max_rows = min(len(subset), 500)
    sample_df = subset.sample(n=max_rows, random_state=None) if len(subset) > max_rows else subset

    out: List[Dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        if len(out) >= limit:
            break
        title = str(row.get("title", ""))
        body = str(row.get("body", ""))
        concept = row.get("concept")
        annotation = row.get("FRAMING_CLASS")
        source = row.get("source")
        analysis = a.analyze_article(title, body)

        if requested_label:
            ord_label = str(analysis.get("ordinal_analysis", {}).get("predicted_label", "")).lower()
            cls_label = str(analysis.get("classification_analysis", {}).get("predicted_label", "")).lower()
            if requested_label not in (ord_label, cls_label):
                continue

        out.append(
            {
                "title": title,
                "body_preview": body[:400],
                "body_full": body,
                "concept": concept,
                "annotation": annotation,
                "source": source,
                "analysis": analysis,
            }
        )

    return _sanitize_for_json({"results": out})


# Convenience root
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Ethoscore API. See /docs for Swagger UI."}


