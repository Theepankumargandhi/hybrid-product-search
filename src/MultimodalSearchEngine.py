import os
import time
import faiss
import torch
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image, ImageFilter
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import streamlit as st

class MultimodalSearchEngine:
    def __init__(self, text_model_name='all-MiniLM-L6-v2', image_model_name='clip-ViT-B-32'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = SentenceTransformer(text_model_name, device=self.device)
        self.image_model = SentenceTransformer(image_model_name, device=self.device)
        self.text_index = None
        self.text_metadata = []
        self.text_bm25_index = None
        self.tokenized_texts = []
        self.image_index = None
        self.image_metadata = []

    def load_text_index(self, index_file='faiss_index_unified.faiss', metadata_file='unified_text_image.csv'):
        self.text_index = faiss.read_index(index_file)
        self.text_metadata = pd.read_csv(metadata_file)
        self.tokenized_texts = [str(text).split() for text in self.text_metadata['raw_text']]
        self.text_bm25_index = BM25Okapi(self.tokenized_texts)

    def load_image_index(self, index_file='image_vector_index_l2.faiss', metadata_file='image_metadata.csv'):
        self.image_index = faiss.read_index(index_file)
        self.image_metadata = pd.read_csv(metadata_file).to_dict(orient='records')

    def search(self, query_input, top_k=5, alpha=0.5):
        if isinstance(query_input, str) and not os.path.isfile(query_input) and not query_input.startswith('http'):
            return self._search_text(query_input, top_k, alpha)
        else:
            return self._search_image(query_input, top_k)

    def _search_text(self, query, top_k=5, alpha=0.5):
        clean_query = query.strip().lower().translate(str.maketrans('', '', '\'"!?;:'))
        query_embedding = self.text_model.encode([clean_query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        semantic_scores, semantic_ids = self.text_index.search(query_embedding, top_k * 2)
        semantic_scores_norm = (semantic_scores + 1) / 2
        tokenized_query = clean_query.split()
        keyword_scores = self.text_bm25_index.get_scores(tokenized_query)
        max_keyword = np.max(keyword_scores)
        keyword_scores_norm = keyword_scores / max_keyword if max_keyword > 0 else keyword_scores
        combined_scores = {}
        distances_map = {}
        for i, idx in enumerate(semantic_ids[0]):
            combined_score = alpha * semantic_scores_norm[0][i] + (1 - alpha) * keyword_scores_norm[idx]
            combined_scores[idx] = combined_score
            distances_map[idx] = 1 - semantic_scores_norm[0][i]
        for idx in np.argsort(keyword_scores_norm)[-top_k * 2:][::-1]:
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * keyword_scores_norm[idx]
        max_combined = max(combined_scores.values()) if combined_scores else 1
        for key in combined_scores:
            combined_scores[key] /= max_combined
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        result_indices = [idx for idx, _ in sorted_ids]
        results = self.text_metadata.iloc[result_indices].copy()
        results['similarity_score'] = [round(score, 2) for _, score in sorted_ids]
        results['distance'] = [round(distances_map.get(idx, np.nan), 4) for idx in result_indices]
        if 'image_url' in results.columns:
            results = results.rename(columns={'image_url': 'product_link'})
        else:
            results['product_link'] = ''
        results['product_link_display'] = results['product_link'].apply(lambda x: f"[🔗 View Product Link]({x})" if pd.notna(x) and str(x).startswith("http") else "")
        results = results[['title', 'distance', 'similarity_score', 'product_link', 'product_link_display']]
        results = results.reset_index(drop=True)
        results.index += 1
        return results

    def _search_image(self, query_input, top_k=5):
        image = self._load_image(query_input)
        if image is None:
            return None
        query_embedding = self.image_model.encode([image], convert_to_numpy=True)
        query_embedding = normalize(query_embedding, norm='l2', axis=1)
        distances, indices = self.image_index.search(query_embedding, top_k)
        results = [self.image_metadata[i] for i in indices[0]]
        results_df = pd.DataFrame(results)
        results_df['distance'] = distances[0].round(4)
        results_df['similarity_score'] = (100 / (1 + distances[0])).round(2)
        results_df = results_df[['title', 'distance', 'similarity_score', 'image_url']]
        results_df = results_df.rename(columns={'image_url': 'product_link'})
        results_df['product_link_display'] = results_df['product_link'].apply(lambda x: f"[🔗 View Product Link]({x})" if pd.notna(x) and str(x).startswith("http") else "")
        results_df = results_df.reset_index(drop=True)
        results_df.index += 1
        return results_df

    def _load_image(self, query_input):
        try:
            if isinstance(query_input, str):
                if query_input.lower().startswith('c:/') or query_input.lower().startswith('d:/'):
                    return None
                if query_input.startswith('http') and not any(ext in query_input.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    query_input = self._extract_image_url_from_product_page(query_input)
                    if query_input is None:
                        return None
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                response = requests.get(query_input, headers=headers, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert('RGB')
                return self._enhance_image_if_needed(image)
            elif hasattr(query_input, 'read'):
                image = Image.open(query_input).convert('RGB')
                return self._enhance_image_if_needed(image)
            else:
                return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def _enhance_image_if_needed(self, image):
        laplacian_var = np.var(np.array(image.filter(ImageFilter.FIND_EDGES)))
        if laplacian_var > 40:
            return image
        elif 10 <= laplacian_var <= 40:
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        else:
            return image

    def _extract_image_url_from_product_page(self, product_url, wait_time=10):
        try:
            if product_url.startswith("http://"):
                product_url = product_url.replace("http://", "https://")
            options = uc.ChromeOptions()
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = uc.Chrome(options=options)
            driver.get(product_url)
            time.sleep(wait_time)
            try:
                og_image = driver.find_element(By.XPATH, "//meta[@property='og:image']")
                og_image_url = og_image.get_attribute("content")
                if og_image_url and any(ext in og_image_url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    driver.quit()
                    return og_image_url
            except:
                pass
            try:
                main_img = driver.find_element(By.CLASS_NAME, 'prod-hero-image-image')
                src = main_img.get_attribute('src')
                if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    driver.quit()
                    return src
            except:
                pass
            try:
                images = driver.find_elements(By.TAG_NAME, 'img')
                selected_img_url = None
                max_resolution = 0
                for img in images:
                    src = img.get_attribute('src')
                    width = int(img.get_attribute('width') or 0)
                    height = int(img.get_attribute('height') or 0)
                    resolution = width * height
                    if src and any(ext in src.lower() for ext in ['.jpg', '.jpeg', '.png']) and width > 300 and height > 300:
                        if resolution > max_resolution:
                            max_resolution = resolution
                            selected_img_url = src
                driver.quit()
                return selected_img_url
            except:
                driver.quit()
                return None
        except Exception as e:
            print(f"Selenium scraping failed: {e}")
            return None
