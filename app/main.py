# Ai4See X-Ray Search Engine - v9.0.0 Deep Numpy Fix
import streamlit as st
import pandas as pd
import os
from PIL import Image
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search_engine import XRaySearchEngine

st.set_page_config(page_title="Ai4See X-Ray Search", layout="wide")

st.title("🔍 Ai4See X-Ray Image Search Interface")
st.markdown("Search across 500+ X-ray images using text descriptions or image uploads.")

@st.cache_resource
def load_engine():
    engine = XRaySearchEngine()
    if os.path.exists("data/index.npy") and os.path.exists("data/metadata.csv"):
        engine.load_index("data/index.npy", "data/metadata.csv")
    return engine

engine = load_engine()

# Sidebar
st.sidebar.header("Dataset Overview")
if engine.metadata is not None:
    st.sidebar.write(f"Total Images: {len(engine.metadata)}")
    st.sidebar.write("Categories:")
    st.sidebar.write(engine.metadata['category'].value_counts())

# Search Tabs
tab1, tab2 = st.tabs(["Text Search", "Image Search"])

with tab1:
    query = st.text_input("Enter search query (e.g., 'chest x-ray', 'fracture', 'dental')", "")
    if query:
        with st.spinner("Searching..."):
            results, scores = engine.search_by_text(query, top_k=20)
            
            # Category Filter (Bonus)
            categories = results['category'].unique().tolist()
            selected_cat = st.multiselect("Filter by category", categories, default=categories)
            
            filtered_results = results[results['category'].isin(selected_cat)]
            filtered_scores = scores[results['category'].isin(selected_cat)]
            
            cols = st.columns(5)
            for i, (idx, row) in enumerate(filtered_results.iterrows()):
                with cols[i % 5]:
                    img_path = os.path.join("data/images", row['image_name'])
                    img = Image.open(img_path)
                    st.image(img, caption=f"{row['category']} (Score: {filtered_scores[i]:.2f})")

with tab2:
    uploaded_file = st.file_uploader("Upload an X-ray image to find similar ones", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Save temp file
        temp_path = "data/temp_query.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.write("Query Image:")
        st.image(Image.open(temp_path), width=200)
        
        with st.spinner("Finding similar images..."):
            results, scores = engine.search_by_image(temp_path, top_k=10)
            
            st.write("Results:")
            cols = st.columns(5)
            for i, (idx, row) in enumerate(results.iterrows()):
                with cols[i % 5]:
                    img_path = os.path.join("data/images", row['image_name'])
                    img = Image.open(img_path)
                    st.image(img, caption=f"{row['category']} (Score: {scores[i]:.2f})")

if engine.metadata is None:
    st.warning("Search engine not indexed yet. Please run the indexing script first.")
