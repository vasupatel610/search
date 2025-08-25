# fashion_search_engine.py
# pip install -U pandas numpy sentence-transformers scikit-learn unidecode fastapi uvicorn

import os
import re
import json
import math
import uuid
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Sequence, Dict, Any, Optional
from unidecode import unidecode
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# =========
# Config
# =========
CSV_PATH = "/home/artisans15/projects/fashion_retail_analytics/data/raw/fashion_products_100_clean.csv"  # <-- point to your CSV
EMB_PATH = "embeddings.npy"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, strong baseline
TOP_K_DEFAULT = 3  # Changed to return top 3 results

# Known taxonomies (extend freely)
KNOWN_TYPES = {
    # footwear
    "sneakers": ["sneaker", "sneakers", "trainers", "running shoes", "running", "jogging"],
    "heels": ["heels", "stiletto", "stilettos", "block heels", "pumps"],
    "boots": ["boots", "ankle boots", "chelsea"],
    "loafers": ["loafer", "loafers"],
    "sandals": ["sandal", "sandals"],
    "slippers": ["slippers", "flip flops"],
    # bags (accessories)
    "handbag": ["handbag", "tote", "shoulder bag"],
    "wallet": ["wallet"],
    "backpack": ["backpack", "rucksack", "laptop backpack"],
    "messenger bag": ["messenger bag", "office bag"],
    "laptop bag": ["laptop bag", "briefcase"],
    # clothing (outerwear, tops, etc.)
    "jacket": ["jacket", "parka", "windcheater"],
    "puffer jacket": ["puffer", "puffer jacket"],
    "sweater": ["sweater", "pullover", "jumper"],
    "hoodie": ["hoodie"],
    "t-shirt": ["t-shirt", "tee"],
    "shirt": ["shirt"],
    "dress": ["dress", "gown"],
    "jeans": ["jeans", "denim jeans"],
    "kurta": ["kurta"],
    "saree": ["saree"],
    "skirt": ["skirt"],
}

OCCASION_HINTS = {
    "party": ["party", "evening", "wedding", "festive", "cocktail", "sparkly", "glitter"],
    "work": ["work", "office", "formal", "business"],
    "casual": ["casual", "everyday", "daily", "weekend"],
    "sports": ["sports", "running", "training", "gym", "athleisure"],
    "winter": ["winter", "warm", "thermal", "puffer", "insulated"],
    "travel": ["travel", "commute", "laptop"],
}

AGE_HINTS = {
    "kids": ["kid", "kids", "children", "child", "boy", "girl", "boys", "girls"],
    "adults": ["men", "women", "adult"],
}

COLOR_SYNONYMS = {
    "navy": "blue",
    "crimson": "red",
    "scarlet": "red",
    "charcoal": "grey",
    "ash": "grey",
    "ivory": "white",
    "beige": "brown",
    "tan": "brown",
    "teal": "green",
}

MATERIAL_SYNONYMS = {
    "pu": "synthetic",
    "faux leather": "synthetic",
    "poly": "polyester",
}

# Only relevant materials per category
CATEGORY_MATERIALS = {
    "Clothing": {"cotton","polyester","denim","silk","wool"},
    "Footwear": {"leather","synthetic","polyester","canvas","suede"},
    "Accessories": {"leather","synthetic","cotton","wool","metal"},
}


# =========
# Utilities
# =========

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unidecode(s)
    s = s.lower().strip()
    # normalize synonyms
    for k,v in COLOR_SYNONYMS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    for k,v in MATERIAL_SYNONYMS.items():
        s = re.sub(rf"\b{k}\b", v, s)
    return s

def infer_product_type(text: str) -> str:
    t = normalize_text(text)
    best_label, best_hitlen = None, 0
    for label, variants in KNOWN_TYPES.items():
        for w in variants:
            if re.search(rf"\b{re.escape(w)}\b", t):
                if len(w) > best_hitlen:
                    best_label, best_hitlen = label, len(w)
    return best_label or ""

def infer_occasion(text: str) -> list:
    t = normalize_text(text)
    tags = []
    for occ, hints in OCCASION_HINTS.items():
        if any(re.search(rf"\b{re.escape(h)}\b", t) for h in hints):
            tags.append(occ)
    return list(set(tags))

def infer_age_group(text: str) -> str:
    t = normalize_text(text)
    for age, hints in AGE_HINTS.items():
        if any(re.search(rf"\b{re.escape(h)}\b", t) for h in hints):
            return "Kids" if age == "kids" else "Adults"
    return ""  # unknown

def coerce_material(category: str, material: str) -> str:
    m = normalize_text(material)
    # synonym map applied in normalize_text, now enforce category validity
    # pick first valid if not allowed
    allowed = CATEGORY_MATERIALS.get(category, set())
    if not allowed:
        return material
    if m in allowed:
        return material
    # fallback: choose a plausible default
    fallback = next(iter(allowed))
    return fallback.capitalize()

def build_search_document(row: pd.Series) -> str:
    # Rich doc used for embedding
    pieces = [
        row.get("product_name", ""),
        row.get("product_description", ""),
        f"category: {row.get('product_category','')}",
        f"brand: {row.get('brand','')}",
        f"type: {row.get('product_type','')}",
        f"color: {row.get('color','')}",
        f"material: {row.get('material','')}",
        f"occasion: {', '.join(row.get('occasion', [])) if isinstance(row.get('occasion'), list) else row.get('occasion','')}",
        f"age: {row.get('age_group','')}",
    ]
    # Add synonym expansions
    expansions = []
    color = normalize_text(str(row.get("color","")))
    if color in COLOR_SYNONYMS.values():
        # add inverse synonyms (e.g., blue -> navy) lightly
        for k, v in COLOR_SYNONYMS.items():
            if v == color:
                expansions.append(k)
    # product-type synonym expansion
    ptype = row.get("product_type","")
    if ptype in KNOWN_TYPES:
        expansions.extend(KNOWN_TYPES[ptype])

    pieces.extend(expansions)
    return " | ".join([p for p in pieces if p])


# =========
# Load data and enrich
# =========

df = pd.read_csv(CSV_PATH)

# Derive optional fields if missing
if "product_type" not in df.columns:
    df["product_type"] = (df["product_name"].fillna("") + " " + df["product_description"].fillna("")).apply(infer_product_type)

if "occasion" not in df.columns:
    df["occasion"] = (df["product_name"].fillna("") + " " + df["product_description"].fillna("")).apply(infer_occasion)

if "age_group" not in df.columns:
    df["age_group"] = (df["product_name"].fillna("") + " " + df["product_description"].fillna("")).apply(infer_age_group)

# Coerce material relevance per category (prevents silk shoes, etc.)
df["material"] = [
    coerce_material(cat, mat) for cat, mat in zip(df["product_category"], df["material"])
]

# Build search documents
df["search_document"] = df.apply(build_search_document, axis=1)

# =========
# Model & Embeddings
# =========

print("Loading model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

def encode_texts(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

if Path(EMB_PATH).exists():
    print("Loading cached embeddings:", EMB_PATH)
    embeddings = np.load(EMB_PATH)
    if embeddings.shape[0] != len(df):
        print("Cached embeddings size mismatch. Recomputing...")
        embeddings = encode_texts(df["search_document"].tolist())
        np.save(EMB_PATH, embeddings)
else:
    print("Computing embeddings...")
    embeddings = encode_texts(df["search_document"].tolist())
    np.save(EMB_PATH, embeddings)

# =========
# Query understanding (naïve facets)
# =========

def parse_query_facets(q: str):
    qn = normalize_text(q)
    facets = {}

    # category hints
    if re.search(r"\b(heel|heels|boot|sneaker|running|loafer|sandals|slippers)\b", qn):
        facets["product_category"] = "Footwear"
    elif re.search(r"\b(handbag|wallet|backpack|messenger|laptop bag|briefcase|belt|watch|sunglasses|scarf)\b", qn):
        facets["product_category"] = "Accessories"
    elif re.search(r"\b(jacket|sweater|hoodie|dress|shirt|t-shirt|kurta|saree|jeans|skirt|parka|puffer)\b", qn):
        facets["product_category"] = "Clothing"

    # product type
    ptype = infer_product_type(qn)
    if ptype:
        facets["product_type"] = ptype

    # occasion
    occ = infer_occasion(qn)
    if occ:
        facets["occasion"] = occ  # list

    # age group
    age = infer_age_group(qn)
    if age:
        facets["age_group"] = age

    # colors present
    # map synonyms to canonical where possible
    color_tokens = set()
    base_colors = set(df["color"].str.lower().unique())
    tokens = re.findall(r"[a-z]+", qn)
    for tok in tokens:
        tok_canon = COLOR_SYNONYMS.get(tok, tok)
        if tok_canon in base_colors:
            color_tokens.add(tok_canon)
    if color_tokens:
        facets["color"] = list(color_tokens)

    return facets

def apply_facets(df_in: pd.DataFrame, facets: dict) -> pd.DataFrame:
    df_out = df_in
    for k, v in facets.items():
        if k not in df_out.columns:
            continue
        if isinstance(v, list):
            df_out = df_out[df_out[k].apply(lambda xs: any(x in xs for x in v) if isinstance(xs, list) else False)]
        else:
            df_out = df_out[df_out[k].astype(str).str.lower() == str(v).lower()]
    return df_out

# =========
# Search & Recommendation APIs
# =========

def semantic_search(query: str, top_k: int = TOP_K_DEFAULT, hybrid_boost: float = 0.2, filter_facets: bool = True):
    """
    Returns top_k matches with scores.
    hybrid_boost: adds a small lexical score (TF-IDF-lite) to re-rank results.
    """
    q_doc = query
    q_emb = encode_texts([q_doc])[0]

    # cosine similarities
    cos_scores = util.cos_sim(q_emb, embeddings).cpu().numpy().ravel()

    candidate_df = df.copy()
    candidate_df["semantic_score"] = cos_scores

    # naive lexical boost: presence count of query tokens in name/desc
    tokens = set(re.findall(r"[a-z0-9]+", normalize_text(query)))
    def lex_score(row):
        text = normalize_text(row["product_name"] + " " + row["product_description"])
        return sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", text))
    candidate_df["lexical_score"] = candidate_df.apply(lex_score, axis=1)

    # combine
    candidate_df["score"] = candidate_df["semantic_score"] + hybrid_boost * candidate_df["lexical_score"]

    # facet filters inferred from query
    if filter_facets:
        facets = parse_query_facets(query)
        filtered = apply_facets(candidate_df, facets)
        if len(filtered) > 0:
            candidate_df = filtered

    # business rules: in-stock first, slight boost for closer price_final to median for robustness
    median_price = candidate_df["price_final"].median() if len(candidate_df) else 0
    candidate_df["business_boost"] = 0.0
    candidate_df.loc[candidate_df["stock_status"] == 1, "business_boost"] += 0.02
    candidate_df["business_boost"] += -0.000005 * (candidate_df["price_final"] - median_price).abs()
    candidate_df["final_score"] = candidate_df["score"] + candidate_df["business_boost"]

    out = candidate_df.sort_values("final_score", ascending=False).head(top_k)
    keep_cols = [
        "product_id","product_name","product_category","brand","product_type",
        "color","material","price_final","stock_status","occasion","age_group",
        "product_description","final_score"
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = ""
    return out[keep_cols].reset_index(drop=True)

def recommend_similar(product_id: str, top_k: int = TOP_K_DEFAULT):
    idxs = np.where(df["product_id"] == product_id)[0]
    if len(idxs) == 0:
        raise ValueError(f"product_id {product_id} not found")
    i = idxs[0]
    vec = embeddings[i]
    scores = util.cos_sim(vec, embeddings).cpu().numpy().ravel()

    out = df.copy()
    out["score"] = scores
    # same category/type preferred, not identical product
    base_cat = df.loc[i, "product_category"]
    base_type = df.loc[i, "product_type"]
    out["score"] += (out["product_category"] == base_cat) * 0.05
    out["score"] += (out["product_type"] == base_type) * 0.05
    out = out[out["product_id"] != product_id]
    out = out.sort_values("score", ascending=False).head(top_k)

    keep_cols = [
        "product_id","product_name","product_category","brand","product_type",
        "color","material","price_final","stock_status","product_description","score"
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = ""
    return out[keep_cols].reset_index(drop=True)

# =========
# FastAPI App
# =========

app = FastAPI(
    title="Fashion Retail Search Engine",
    description="A semantic search engine for fashion products that returns the top 3 closest recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SearchQuery(BaseModel):
    query: str = Query(..., description="Search query")
    top_k: int = Query(TOP_K_DEFAULT, description="Number of results to return (max 10)")
    hybrid_boost: float = Query(0.2, description="Lexical boost factor")
    filter_facets: bool = Query(True, description="Apply facet filters to results")

class SearchResponse(BaseModel):
    query: str
    results: Sequence[Dict[str, Any]]
    total_results: int
    search_time_ms: float

@app.get("/")
async def root():
    return {
        "message": "Fashion Retail Search Engine API",
        "endpoints": {
            "/search": "Search for products",
            "/recommend/{product_id}": "Get recommendations for a product",
            "/categories": "Get all product categories",
            "/stats": "Get dataset statistics"
        }
    }

@app.get("/search")
async def search(
    query: str = Query(..., description="Search query"),
    top_k: int = Query(TOP_K_DEFAULT, description="Number of results to return (max 10)"),
    hybrid_boost: float = Query(0.2, description="Lexical boost factor"),
    filter_facets: bool = Query(True, description="Apply facet filters to results")
):
    """
    Search for fashion products using semantic search.
    Returns the top 3 closest matches by default.
    """
    import time
    start_time = time.time()
    
    try:
        # Limit top_k to prevent abuse
        top_k = min(max(1, top_k), 10)
        
        results = semantic_search(
            query=query,
            top_k=top_k,
            hybrid_boost=hybrid_boost,
            filter_facets=filter_facets
        )
        
        search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Ensure all keys in results are strings for Pydantic compatibility
        results_records = [
            {str(k): v for k, v in rec.items()}
            for rec in results.to_dict(orient="records")
        ]
        return SearchResponse(
            query=query,
            results=results_records,
            total_results=len(results),
            search_time_ms=round(search_time, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/recommend/{product_id}")
async def recommend(
    product_id: str,
    top_k: int = Query(TOP_K_DEFAULT, description="Number of recommendations to return")
):
    """
    Get product recommendations based on a product ID.
    Returns the top 3 most similar products by default.
    """
    try:
        top_k = min(max(1, top_k), 10)
        recs = recommend_similar(product_id, top_k=top_k)
        return {
            "product_id": product_id,
            "recommendations": recs.to_dict(orient="records"),
            "total_recommendations": len(recs)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Product not found: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

@app.get("/categories")
async def get_categories():
    """Get all available product categories"""
    try:
        categories = df["product_category"].unique().tolist()
        return {"categories": categories, "total": len(categories)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get categories: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get dataset statistics"""
    try:
        stats = {
            "total_products": len(df),
            "categories": df["product_category"].value_counts().to_dict(),
            "brands": df["brand"].nunique(),
            "price_range": {
                "min": float(df["price_final"].min()),
                "max": float(df["price_final"].max()),
                "mean": float(df["price_final"].mean())
            }
        }
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    print("Starting Fashion Retail Search Engine...")
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {len(df)} products")
    print("API available at: http://localhost:8000")
    print("API docs at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
