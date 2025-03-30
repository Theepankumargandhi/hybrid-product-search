# 🧠 Multimodal Product Similarity Search Engine
# hybrid-product-search
Multimodal Product Search using FAISS, CLIP, MiniLM, and BM25

This project is a **Multimodal Product Search Engine** that supports searching via **text**, **image URL**, or **image upload**. It uses a hybrid approach combining **semantic embeddings**, **keyword search (BM25)**, and **image similarity (CLIP + FAISS)** to retrieve top-matching products.

---

## 🚀 Features

- 🔎 Search using **text descriptions**, **image URLs**, or **uploaded images**
- 🤖 Uses **MiniLM** for text embeddings and **CLIP** for image embeddings
- 🧠 Combines **FAISS** for fast similarity search with **BM25** for keyword matching
- 📦 Built with **Streamlit** for a clean interactive UI
- 📸 Automatically extracts product images from URLs (via Selenium scraping)

---

## 📁 Folder Structure

```bash
hybrid-product-search/
│
├── README.md                     # Project documentation
├── requirements.txt              # Dependency list
├── app/
│   └── streamlit_app.py          # Streamlit user interface
├── src/
│   └── MultimodalSearchEngine.py # Core search engine logic
├── notebooks/                    # EDA or experiments (optional)
├── utils/                        # Utility functions (optional)
├── outputs/                      # Diagrams, charts, or results
├── faiss_index_unified.faiss     # Text-image FAISS index
├── image_vector_index_l2.faiss   # Image FAISS index
├── unified_text_image.csv        # Metadata for text queries
└── image_metadata.csv            # Metadata for image queries

📁 Note: The full dataset and embeddings are not included due to size. 
To replicate this project, follow the instructions in `data_preparation.md` (coming soon) or contact me for access.


## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/Theepankumargandhi/hybrid-product-search.git
cd hybrid-product-search

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app/streamlit_app.py


## 💼 Use Cases
- E-commerce product recommendation and search
- Retail product catalog matching
- Visual + semantic product discovery



