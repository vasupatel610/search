# app.py ── Streamlit UI for the FastAPI search service
import streamlit as st
import requests
import pandas as pd

# -------------------------------------------------------------------
# 1.  Configuration
# -------------------------------------------------------------------
API_URL   = st.secrets.get("api_url", "http://localhost:8000")
HEALTH_EP = f"{API_URL}/health"
SEARCH_EP = f"{API_URL}/search"

# -------------------------------------------------------------------
# 2.  Check back-end status once at start-up
# -------------------------------------------------------------------
@st.cache_data(ttl=60)   # refresh every 60 s
def get_backend_status():
    try:
        return requests.get(HEALTH_EP, timeout=5).json()
    except Exception as e:
        return {"status": "unreachable", "detail": str(e)}

status = get_backend_status()
st.sidebar.title("Back-end status")
st.sidebar.write(status)

if status.get("status") != "healthy":
    st.error("FastAPI back-end is not reachable — start it first.")
    st.stop()

# -------------------------------------------------------------------
# 3.  Page layout
# -------------------------------------------------------------------
st.title("🔎 Natural-language product search")

with st.form(key="query_form"):
    query   = st.text_input("Enter search query", placeholder="e.g. blue denim jacket under ₹2,000")
    top_k   = st.number_input("Results", 1, 50, 10, step=1)
    alpha   = st.slider("Semantic weight (α)", 0.0, 1.0, status.get("embedding_method") == "groq" and 0.7 or 0.3, 0.05)
    submitted = st.form_submit_button("Search")

# -------------------------------------------------------------------
# 4.  Call API and display results
# -------------------------------------------------------------------
def search_api(q, k, a):
    payload = {"query": q, "top_k": k, "alpha": a}
    r = requests.post(SEARCH_EP, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()

if submitted:
    if not query.strip():
        st.warning("Please type a query.")
        st.stop()

    with st.spinner("Searching…"):
        try:
            res = search_api(query, int(top_k), float(alpha))
        except Exception as e:
            st.error(f"API call failed: {e}")
            st.stop()

    hits = res["results"]
    if not hits:
        st.info("No matches found.")
    else:
        st.success(f"{res['total_results']} results (method: {res['search_method']})")
        # Build a DataFrame for pretty display
        records = []
        for h in hits:
            record = {
                "Score": f"{h['score']:.3f}",
                "Name":  h["product_name"],
                "Description": h["product_description"][:80] + "…" if len(h["product_description"]) > 80 else h["product_description"],
                "Price": f"₹{h['price']:,.0f}",
            }
            # flatten enhanced fields into their own columns
            if h.get("enhanced_fields"):
                record.update(h["enhanced_fields"])
            records.append(record)

        st.dataframe(pd.DataFrame(records))

        with st.expander("Field analysis (from back-end)"):
            st.json(res.get("field_analysis", {}))
