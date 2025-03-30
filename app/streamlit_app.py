# Streamlit UI for Multimodal Search Engine (Text or Image)

import streamlit as st
import pandas as pd
import time
from PIL import Image
from sentence_transformers import SentenceTransformer
import faiss
from MultimodalSearchEngine import MultimodalSearchEngine

# Set Streamlit page config
st.set_page_config(page_title="Multimodal Product Search", layout="wide")

# Title and Description
st.title("Multimodal Product Similarity Search")
st.markdown("Search for products using either Text or Image (URL or Upload).")

# Load search engine and models
@st.cache_resource
def load_engine():
    engine = MultimodalSearchEngine()
    engine.load_text_index("faiss_index_unified.faiss", "unified_text_image.csv")
    engine.load_image_index("image_vector_index_l2.faiss", "image_metadata.csv")
    return engine

engine = load_engine()

# Input Type Selector
query_mode = st.radio("Select Query Type", ["Text Input", "Image URL", "Upload Image"])
query_input = None

if query_mode == "Text Input":
    query_input = st.text_input("Enter your text query")
    alpha = st.slider("Weight: Semantic vs Keyword Score (Alpha)", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    use_alpha = True

elif query_mode == "Image URL":
    query_input = st.text_input("Paste image URL")
    use_alpha = False

elif query_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        query_input = uploaded_file
        st.image(query_input, caption="Uploaded image", use_container_width=True)
    use_alpha = False

# Search Button
if st.button("Search Similar Products"):
    if query_input:
        start_time = time.time()

        with st.spinner("Searching..."):
            if query_mode == "Text Input" and use_alpha:
                results = engine.search(query_input=query_input, top_k=10, alpha=alpha)
            else:
                results = engine.search(query_input=query_input, top_k=10)

        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)

        if results is not None and not results.empty:
            st.success(f"Search completed in {elapsed_time} seconds")

            for i in range(len(results)):
                row = results.iloc[i]
                cols = st.columns([1, 4, 1])  # Image | Title + Link | Score

                with cols[0]:
                    try:
                        st.image(row['product_link'], width=130)
                    except:
                        st.write("Image not available")

                with cols[1]:
                    st.markdown(f"**{row['title']}**")
                    if pd.notna(row.get('product_link_display')):
                        st.markdown(row['product_link_display'])

                with cols[2]:
                    st.markdown(f"Score: {row['similarity_score']}")

                st.markdown("---")
        else:
            st.warning("No similar products found.")
    else:
        st.error("Please enter a valid input.")
