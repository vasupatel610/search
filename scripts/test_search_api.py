"""
scripts/test_search_api.py
--------------------------

Hybrid product-search API with Groq LLM for intelligent field selection + SBERT score fusion.
Run: uvicorn scripts.test_search_api:app --reload --port 8000
"""

import os
from typing import List, Optional, Dict, Any
import time
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import numpy as np
import httpx
import logging, sys
from functools import wraps
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import json

# ── 1. Global logging format ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("search-api")

# ── 2. Decorator to log function entry/exit and duration ────────────────────
def log_calls(name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.debug(f"{name} → called")
            out = fn(*args, **kwargs)
            logger.debug(f"{name} → completed in {time.time()-start:.3f}s")
            return out
        return wrapper
    return decorator

# ── 3. FastAPI middleware for per-request logging ───────────────────────────
class LogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"⇢ {request.method} {request.url.path}")
        resp = await call_next(request)
        logger.info(f"⇠ {request.method} {request.url.path} → {resp.status_code}")
        return resp

################################################################################
# Configuration
################################################################################

# Groq API
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")
GROQ_MODEL = "llama3-8b-8192"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"

# SBERT model for lexical matching
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Fusion weight: 1.0 → only semantic, 0.0 → only lexical
ALPHA = 0.7

# Network settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0
TIMEOUT = 30.0

################################################################################
# Load models once at start-up
################################################################################

def load_sbert_model():
    """Load SBERT model with error handling"""
    try:
        logger.info("Loading SBERT model...")
        model = SentenceTransformer(SBERT_MODEL_NAME)
        logger.info("SBERT model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load SBERT model: {e}")
        raise

sbert = load_sbert_model()

################################################################################
# Groq LLM Field Selection
################################################################################

class GroqFieldSelector:
    """Uses Groq LLM to intelligently select relevant fields from CSV for search enhancement"""
    
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
    
    def analyze_fields(self, df: pd.DataFrame, sample_size: int = 5) -> Dict[str, Any]:
        """
        Analyze CSV fields and determine which ones are useful for product search
        """
        # Get sample data for analysis
        sample_df = df.head(sample_size)
        
        # Create field analysis prompt
        field_info = {}
        for col in df.columns:
            if col in ['product_name', 'product_description', 'price']:
                continue  # Skip required fields
                
            # Get sample values and data type
            sample_values = sample_df[col].dropna().head(3).tolist()
            data_type = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            
            field_info[col] = {
                "sample_values": sample_values,
                "data_type": data_type,
                "null_percentage": (null_count / len(df)) * 100,
                "unique_count": unique_count,
                "total_records": len(df)
            }
        
        prompt = self._create_field_analysis_prompt(field_info)
        
        try:
            response = self._call_groq_chat(prompt)
            analysis = self._parse_groq_response(response)
            logger.info(f"Groq field analysis completed. Selected fields: {analysis.get('selected_fields', [])}")
            return analysis
        except Exception as e:
            logger.warning(f"Groq field analysis failed: {e}")
            # Fallback to heuristic selection
            return self._fallback_field_selection(df)
    
    def _create_field_analysis_prompt(self, field_info: Dict) -> str:
        """Create prompt for Groq to analyze fields"""
        
        prompt = """You are an expert data scientist analyzing a product dataset for search optimization. 
Your task is to identify which fields (columns) would be most valuable for enhancing product search beyond the basic required fields (product_name, product_description, price).

Dataset Fields Analysis:
"""
        
        for field, info in field_info.items():
            prompt += f"""
Field: {field}
- Data Type: {info['data_type']}
- Sample Values: {info['sample_values']}
- Unique Values: {info['unique_count']} out of {info['total_records']} records
- Missing Data: {info['null_percentage']:.1f}%
"""
        
        prompt += """
Please analyze these fields and return a JSON response with the following structure:
{
    "selected_fields": ["field1", "field2", ...],
    "field_weights": {"field1": 0.8, "field2": 0.6, ...},
    "field_purposes": {"field1": "category classification", "field2": "brand matching", ...},
    "reasoning": "Brief explanation of why these fields were selected"
}

Selection Criteria:
1. Fields that help with product categorization (category, type, style, etc.)
2. Fields for brand/manufacturer identification
3. Fields with descriptive attributes (color, size, material, etc.)
4. Fields with reasonable data quality (not too many nulls)
5. Avoid ID fields, timestamps, or purely numeric codes without meaning

Weight the fields from 0.1 to 1.0 based on their expected search relevance.
Return only valid JSON, no additional text."""
        
        return prompt
    
    def _call_groq_chat(self, prompt: str) -> str:
        """Call Groq chat completion API"""
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        with httpx.Client(timeout=TIMEOUT) as client:
            for attempt in range(MAX_RETRIES):
                try:
                    logger.debug(f"Groq chat API call - attempt {attempt + 1}")
                    r = client.post(GROQ_CHAT_URL, headers=self.headers, json=payload)
                    
                    if r.status_code == 200:
                        response = r.json()
                        return response["choices"][0]["message"]["content"]
                    else:
                        logger.warning(f"Groq API returned {r.status_code}: {r.text}")
                        if attempt == MAX_RETRIES - 1:
                            raise RuntimeError(f"Groq API failed: {r.text}")
                
                except (httpx.ConnectError, httpx.TimeoutException) as e:
                    logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                    if attempt == MAX_RETRIES - 1:
                        raise RuntimeError(f"Network error: {e}")
                    time.sleep(RETRY_DELAY * (attempt + 1))
        
        raise RuntimeError("Groq API call failed after all retries")
    
    def _parse_groq_response(self, response: str) -> Dict[str, Any]:
        """Parse Groq JSON response"""
        try:
            # Extract JSON from response (in case there's extra text)
            start = response.find('{')
            end = response.rfind('}') + 1
            json_str = response[start:end]
            
            parsed = json.loads(json_str)
            
            # Validate required fields
            required_keys = ['selected_fields', 'field_weights', 'field_purposes', 'reasoning']
            if not all(key in parsed for key in required_keys):
                raise ValueError("Missing required keys in Groq response")
            
            return parsed
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse Groq response: {e}")
            logger.error(f"Raw response: {response}")
            raise
    
    def _fallback_field_selection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fallback heuristic field selection if Groq fails"""
        logger.info("Using fallback heuristic field selection")
        
        # Common field patterns that are usually useful for search
        useful_patterns = [
            'category', 'brand', 'color', 'size', 'material', 'type', 
            'style', 'gender', 'age', 'season', 'occasion', 'pattern',
            'fabric', 'fit', 'sleeve', 'neck', 'collar', 'closure'
        ]
        
        selected_fields = []
        field_weights = {}
        field_purposes = {}
        
        for col in df.columns:
            if col in ['product_name', 'product_description', 'price']:
                continue
            
            col_lower = col.lower()
            # Check if column name contains useful patterns
            for pattern in useful_patterns:
                if pattern in col_lower:
                    selected_fields.append(col)
                    field_weights[col] = 0.6  # Default weight
                    field_purposes[col] = f"heuristic match for {pattern}"
                    break
            
            # Also include fields with reasonable cardinality and low null rate
            if col not in selected_fields:
                null_rate = df[col].isnull().sum() / len(df)
                unique_count = df[col].nunique()
                
                if (null_rate < 0.7 and 
                    2 <= unique_count <= len(df) * 0.5 and 
                    df[col].dtype in ['object', 'string']):
                    selected_fields.append(col)
                    field_weights[col] = 0.4
                    field_purposes[col] = "heuristic selection based on data quality"
        
        return {
            "selected_fields": selected_fields[:8],  # Limit to top 8 fields
            "field_weights": field_weights,
            "field_purposes": field_purposes,
            "reasoning": "Fallback heuristic selection based on common field patterns and data quality"
        }

################################################################################
# Enhanced Index with Field Selection
################################################################################

class Index:
    """
    Enhanced index with Groq-selected fields for better search
    """
    def __init__(self, df: pd.DataFrame, use_groq_embeddings: bool = True):
        self.df = df.reset_index(drop=True)
        self.use_groq_embeddings = use_groq_embeddings
        
        # Use Groq to select relevant fields
        logger.info("Analyzing dataset fields with Groq LLM...")
        field_selector = GroqFieldSelector()
        self.field_analysis = field_selector.analyze_fields(df)
        
        # Create enhanced text combining selected fields
        self.selected_fields = self.field_analysis['selected_fields']
        self.field_weights = self.field_analysis['field_weights']
        
        logger.info(f"Selected fields for search enhancement: {self.selected_fields}")
        
        combined_text = self._create_enhanced_text()
        
        # Create embeddings
        if use_groq_embeddings:
            try:
                logger.info("Creating Groq embeddings...")
                self.embeddings = self._embed_texts_groq(combined_text)
                logger.info("Groq embeddings created successfully")
            except Exception as e:
                logger.warning(f"Groq embeddings failed: {e}")
                logger.info("Falling back to SBERT for semantic embeddings...")
                self.embeddings = self._embed_texts_sbert(combined_text)
                self.use_groq_embeddings = False
        else:
            logger.info("Using SBERT for semantic embeddings...")
            self.embeddings = self._embed_texts_sbert(combined_text)

        # Lexical embeddings using SBERT
        logger.info("Creating lexical embeddings...")
        self.term_vecs = sbert.encode(
            combined_text,
            convert_to_tensor=True, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        logger.info("Enhanced index initialization complete")
    
    def _create_enhanced_text(self) -> List[str]:
        """Create enhanced text by combining selected fields with weights"""
        enhanced_texts = []
        
        for _, row in self.df.iterrows():
            # Start with required fields
            text_parts = [
                str(row["product_name"]),
                str(row["product_description"])
            ]
            
            # Add selected fields with repetition based on weights
            for field in self.selected_fields:
                if field in row and pd.notna(row[field]):
                    field_value = str(row[field])
                    weight = self.field_weights.get(field, 0.5)
                    
                    # Repeat field based on weight (higher weight = more repetitions)
                    repetitions = max(1, int(weight * 3))
                    text_parts.extend([field_value] * repetitions)
            
            enhanced_texts.append(" ".join(text_parts))
        
        return enhanced_texts

    def _embed_texts_groq(self, texts: List[str]) -> np.ndarray:
        """Call Groq embeddings endpoint (batched) and return ndarray (N, dim)."""
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        out = []
        batch_size = 32
        
        with httpx.Client(timeout=TIMEOUT) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": "text-embedding-ada-002",
                    "input": batch
                }
                
                for attempt in range(MAX_RETRIES):
                    try:
                        logger.debug(f"Groq API call - batch {i//batch_size + 1}, attempt {attempt + 1}")
                        r = client.post(GROQ_EMBED_URL, headers=headers, json=payload)
                        
                        if r.status_code == 200:
                            data = r.json()["data"]
                            out.extend([d["embedding"] for d in data])
                            break
                        else:
                            logger.warning(f"Groq API returned {r.status_code}: {r.text}")
                            if attempt == MAX_RETRIES - 1:
                                raise RuntimeError(f"Groq API failed: {r.text}")
                            
                    except (httpx.ConnectError, httpx.TimeoutException) as e:
                        logger.warning(f"Network error on attempt {attempt + 1}: {e}")
                        if attempt == MAX_RETRIES - 1:
                            raise RuntimeError(f"Network error: {e}")
                        time.sleep(RETRY_DELAY * (attempt + 1))
                
                time.sleep(0.1)  # Rate limiting
                
        return np.array(out, dtype=np.float32)

    def _embed_texts_sbert(self, texts: List[str]) -> np.ndarray:
        """Use SBERT for embeddings as fallback"""
        embeddings = sbert.encode(
            texts, 
            convert_to_tensor=False, 
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return np.array(embeddings, dtype=np.float32)

    def search(self, query: str, top_k: int = 10):
        """Enhanced search using selected fields"""
        # Create enhanced query text (could be improved with field-aware query processing)
        enhanced_query = query  # For now, use query as-is
        
        # Semantic embeddings
        if self.use_groq_embeddings:
            try:
                q_emb_sem = self._embed_texts_groq([enhanced_query])[0]
            except Exception as e:
                logger.warning(f"Groq embedding failed for query, using SBERT: {e}")
                q_emb_sem = self._embed_texts_sbert([enhanced_query])[0]
        else:
            q_emb_sem = self._embed_texts_sbert([enhanced_query])[0]
            
        # Calculate semantic similarity
        sem_scores = self.embeddings @ q_emb_sem / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb_sem) + 1e-9
        )

        # Lexical similarity
        q_emb_lex = sbert.encode(enhanced_query, convert_to_tensor=True, normalize_embeddings=True)
        lex_scores = util.dot_score(self.term_vecs, q_emb_lex).squeeze().numpy()

        # Weighted score
        final = ALPHA * sem_scores + (1 - ALPHA) * lex_scores
        idx = final.argsort()[::-1][:top_k]
        return idx, final[idx]

################################################################################
# FastAPI Models
################################################################################

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    alpha: Optional[float] = None

class SearchHit(BaseModel):
    product_name: str
    product_description: str
    price: float
    score: float
    enhanced_fields: Optional[Dict[str, Any]] = None  # Include selected field values

class SearchResponse(BaseModel):
    results: List[SearchHit]
    total_results: int
    search_method: str
    field_analysis: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    index_ready: bool
    embedding_method: str
    selected_fields: Optional[List[str]] = None

################################################################################
# FastAPI App
################################################################################

app = FastAPI(title="Enhanced Hybrid Product Search API", version="2.0.0")
app.add_middleware(LogMiddleware)

# Global index
INDEX: Optional[Index] = None

def build_index_from_file(path: str, use_groq: bool = True) -> Index:
    """Build enhanced index from file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif ext == ".json":
            df = pd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    except Exception as e:
        raise ValueError(f"Failed to read file {path}: {e}")

    # Ensure compulsory fields
    required = {"product_name", "product_description", "price"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Clean data
    df = df.dropna(subset=list(required))
    
    logger.info(f"Building enhanced index with {len(df)} products")
    return Index(df, use_groq_embeddings=use_groq)

def initialize_index():
    """Initialize the enhanced search index"""
    global INDEX
    
    dataset_path = os.getenv("DATASET_PATH", "/home/artisans15/projects/fashion_retail_analytics/data/raw/myntra_products_catalog.csv")
    
    try:
        logger.info(f"Initializing enhanced index from: {dataset_path}")
        INDEX = build_index_from_file(dataset_path, use_groq=True)
        logger.info("Enhanced index built successfully")
    except Exception as e:
        logger.warning(f"Failed to build enhanced index with Groq: {e}")
        try:
            logger.info("Attempting to build enhanced index with SBERT only...")
            INDEX = build_index_from_file(dataset_path, use_groq=False)
            logger.info("Enhanced index built successfully with SBERT")
        except Exception as e2:
            logger.error(f"Failed to build enhanced index: {e2}")
            INDEX = None

# Initialize on startup
initialize_index()

################################################################################
# API Routes
################################################################################

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Enhanced health check endpoint"""
    embedding_method = "none"
    selected_fields = None
    
    if INDEX:
        embedding_method = "groq" if INDEX.use_groq_embeddings else "sbert"
        selected_fields = INDEX.selected_fields
    
    return HealthResponse(
        status="healthy" if INDEX else "unhealthy",
        index_ready=INDEX is not None,
        embedding_method=embedding_method,
        selected_fields=selected_fields
    )

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Enhanced search with field analysis"""
    global ALPHA
    
    if INDEX is None:
        raise HTTPException(
            status_code=503, 
            detail="Search index not available. Check /health endpoint for status."
        )

    # Allow per-request alpha override
    old_alpha = ALPHA
    if req.alpha is not None:
        ALPHA = max(0.0, min(1.0, req.alpha))

    try:
        top_k = min(req.top_k or 10, len(INDEX.df))
        idxs, scores = INDEX.search(req.query, top_k)
        
        hits = []
        for i, sc in zip(idxs, scores):
            row = INDEX.df.iloc[i]
            
            # Include enhanced field values
            enhanced_fields = {}
            for field in INDEX.selected_fields:
                if field in row.index and pd.notna(row[field]):
                    enhanced_fields[field] = str(row[field])
            
            hits.append(
                SearchHit(
                    product_name=str(row["product_name"]),
                    product_description=str(row["product_description"]),
                    price=float(row["price"]),
                    score=float(sc),
                    enhanced_fields=enhanced_fields
                )
            )
        
        search_method = "enhanced_hybrid_groq" if INDEX.use_groq_embeddings else "enhanced_hybrid_sbert"
        
        return SearchResponse(
            results=hits,
            total_results=len(hits),
            search_method=search_method,
            field_analysis={
                "selected_fields": INDEX.selected_fields,
                "field_purposes": INDEX.field_analysis.get('field_purposes', {}),
                "reasoning": INDEX.field_analysis.get('reasoning', '')
            }
        )
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    finally:
        ALPHA = old_alpha

@app.get("/field-analysis")
def get_field_analysis():
    """Get detailed field analysis information"""
    if INDEX is None:
        raise HTTPException(
            status_code=503, 
            detail="Search index not available."
        )
    
    return {
        "field_analysis": INDEX.field_analysis,
        "dataset_columns": list(INDEX.df.columns),
        "selected_fields": INDEX.selected_fields,
        "dataset_shape": INDEX.df.shape
    }

@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "Enhanced Hybrid Product Search API with Groq Field Selection",
        "version": "2.0.0",
        "features": [
            "Groq LLM for intelligent field selection",
            "Enhanced search with weighted field importance",
            "Hybrid semantic + lexical search",
            "Fallback mechanisms for reliability"
        ],
        "endpoints": {
            "health": "/health",
            "search": "/search",
            "field-analysis": "/field-analysis",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    