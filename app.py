import streamlit as st
from sentence_transformers import SentenceTransformer, CrossEncoder
from ddgs import DDGS

from search.chroma_store import load_chroma_collection
from search.data_loader import load_raw_data
from search.dense import dense_search
from search.bm25 import bm25_search
from search.hybrid import hybrid_rerank

from evaluation.metrics import recall_at_k, mrr, ndcg

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Research Paper Semantic Search",
    layout="wide"
)

st.sidebar.header("Evaluation")

show_metrics = st.sidebar.checkbox(
    "Show Evaluation Metrics",
    help="Uses top-1 dense result as pseudo-relevant document"
)

# -------------------- LOAD RESOURCES (ONCE) --------------------
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return embedder, reranker

@st.cache_resource
def load_backend():
    collection = load_chroma_collection()
    titles, abstracts, bm25 = load_raw_data()
    return collection, titles, abstracts, bm25

embedding_model, reranker = load_models()
collection, titles, abstracts, bm25 = load_backend()

# -------------------- AGENT FUNCTIONS --------------------
def search_paper_on_internet(paper_title):
    """Search for paper on internet using DuckDuckGo"""
    try:
        ddgs = DDGS()
        results = ddgs.text(f"{paper_title} arxiv", max_results=10)
        
        # Filter for arxiv links only
        arxiv_results = [
            r for r in results
            if "arxiv.org" in r.get("href", "")
        ]
        
        # Convert to standard format
        formatted_results = [
            {
                "title": r.get("title", ""),
                "link": r.get("href", ""),
                "snippet": r.get("body", "")
            }
            for r in arxiv_results
        ]
        
        return formatted_results
    except Exception as e:
        st.error(f"Error searching for paper: {str(e)}")
        import traceback
        st.write(traceback.format_exc())
        return None

def convert_arxiv_to_pdf(arxiv_link):
    """Convert arxiv abstract link to PDF link"""
    if arxiv_link and "arxiv.org" in arxiv_link:
        return arxiv_link.replace("/abs/", "/pdf/") + ".pdf"
    return None

# -------------------- UI --------------------
st.image("assets/banner.png")
st.markdown('<h1 style="text-align: center;">🔍 Research Paper Semantic Search Engine</h1>', unsafe_allow_html=True)

query = st.text_input("Enter your research query")

mode = st.radio(
    "Search Mode",
    ["Dense", "BM25", "Hybrid + Re-ranked"]
)

top_k = st.slider("Top K Results", 3, 20, 5)

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = None
if "search_mode" not in st.session_state:
    st.session_state.search_mode = None

if st.button("Search") and query:

    if mode == "Dense":
        results = dense_search(collection, embedding_model, query, top_k)

    elif mode == "BM25":
        results = bm25_search(bm25, titles, abstracts, query, top_k)

    else:
        dense_results = dense_search(collection, embedding_model, query, 10)
        bm25_results = bm25_search(bm25, titles, abstracts, query, 10)
        results = hybrid_rerank(
            query, dense_results, bm25_results, reranker, top_k
        )

    # Store results in session state
    st.session_state.search_results = results
    st.session_state.search_mode = mode

# Display results if they exist in session state
if st.session_state.search_results is not None:
    results = st.session_state.search_results
    mode = st.session_state.search_mode
    
    st.markdown("---")

    for r in results:
        st.markdown(f"### {r['title']}")
        st.write(r["abstract"][:400] + "...")
        st.json({k: v for k, v in r.items() if "score" in k})
    
    # -------- Internet Search Feature (Hybrid + Re-ranked only) --------
    if mode == "Hybrid + Re-ranked":
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Search Internet for Paper", use_container_width=True, key="search_internet_btn"):
                top_result_title = results[0]["title"]
                
                with st.spinner("🔍 Searching for paper on internet..."):
                    arxiv_results = search_paper_on_internet(top_result_title)
                
                if arxiv_results and len(arxiv_results) > 0:
                    st.markdown("---")
                    # Create a nice card-like container
                    with st.container():
                        st.markdown(
                            """
                                <h2 style='margin: 0; font-size: 20px; text-align: center;'>Paper Found in the Internet! ✅</h2>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    paper_link = arxiv_results[0]["link"]
                    pdf_link = convert_arxiv_to_pdf(paper_link)
                    
                    # Paper info card
                    st.markdown(
                        f"""
                            <div style='background: #f0f2f6; padding: 15px; border-radius: 8px; 
                            border-left: 4px solid #0066cc; margin-bottom: 15px;'>
                                <p style='margin: 0; color: #000046; font-size: 16px; font-weight: 500; text-align: center;'>Title:</p>
                                <p style='margin: 8px 0 0 0; color: #0066cc; font-size: 15px; text-align: center;'><b>{arxiv_results[0]['title']}</b></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Open paper button centered
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col2:
                        st.markdown(
                            f"""
                                <a href="{pdf_link}" target="_blank" style='display: inline-block; 
                                background: linear-gradient(135deg, #000046 0%, #1CB5E0 100%);
                                color: white; padding: 12px 30px; border-radius: 8px; 
                                text-decoration: none; font-weight: 600; font-size: 16px; 
                                text-align: center; width: 200px; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                                transition: transform 0.2s, box-shadow 0.2s;'>
                                Open Paper
                                </a>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown("---")
                    st.markdown(
                        """
                        <div style='background: #fee; padding: 15px; border-radius: 8px; 
                        border-left: 4px solid #ff6b6b; text-align: center;'>
                            <p style='margin: 0; color: #c92a2a; font-size: 16px; font-weight: 500;'>
                                ❌ Could not fetch paper from internet
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    
    # ---------------- Evaluation ----------------
    if show_metrics:
        # Pseudo relevance: top-1 dense result
        dense_eval = dense_search(
            collection, embedding_model, query, top_k=1
        )

        # Use abstract text as identifier for relevant/retrieved docs
        relevant_abstract = dense_eval[0]["abstract"]
        retrieved_abstracts = [r["abstract"] for r in results]

        k = len(retrieved_abstracts)

        # For metrics, we need to check if the relevant doc is in retrieved
        # Since we only have 1 relevant doc, convert to list for metric functions
        recall_score = recall_at_k(
            relevant=[relevant_abstract],
            retrieved=retrieved_abstracts,
            k=k
        )

        mrr_score = mrr(
            relevant=[relevant_abstract],
            retrieved=retrieved_abstracts
        )

        ndcg_score = ndcg(
            relevant=[relevant_abstract],
            retrieved=retrieved_abstracts,
            k=k
        )

        st.sidebar.markdown("### 📊 Evaluation Metrics")
        st.sidebar.metric("MRR", f"{mrr_score:.3f}")
        st.sidebar.metric("Recall@K", f"{recall_score:.3f}")
        st.sidebar.metric("NDCG@K", f"{ndcg_score:.3f}")