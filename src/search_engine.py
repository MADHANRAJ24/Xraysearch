import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import os

class XRaySearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Version marker for logs - we use v9.0.0 for the Deep Numpy fix
        print("--- X-Ray Search Engine Version: 9.0.0 (Deep Numpy Fix) ---")
        
        # We stay on CPU for maximum reliability in cloud environments
        self.device = "cpu"
        print(f"Loading search engine on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
        self.embeddings = None
        self.metadata = None

    def _numpy_normalize(self, x):
        """Ultra-stable normalization using basic Numpy arithmetic."""
        # Step 1: Force to a standard numeric numpy array
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        
        # We use asarray with dtype to ensure it's not a 'wrapped' object array
        try:
            x = np.asarray(x, dtype=np.float32)
        except Exception as e:
            print(f"Aggressive conversion triggered: {e}")
            # If x is an object (like a dict or list), try to extract nested values
            if hasattr(x, "values"): x = x.values()
            x = np.array(list(x), dtype=np.float32)

        # Step 2: Ensure 2D shape [1, 512]
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Step 3: Manual L2 Normalization using basic math only
        # This replaces np.linalg.norm for absolute safety
        sq_sum = np.sum(x * x, axis=-1, keepdims=True)
        norm = np.sqrt(sq_sum + 1e-12)
        return x / norm

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        return self._numpy_normalize(outputs)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to("cpu")
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        return self._numpy_normalize(outputs)

    def index_dataset(self, metadata_path, image_dir):
        print("Indexing dataset...")
        self.metadata = pd.read_csv(metadata_path)
        all_embeddings = []
        
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(image_dir, row['image_name'])
            if os.path.exists(img_path):
                emb = self.get_image_embedding(img_path)
                all_embeddings.append(emb)
            else:
                print(f"Warning: Image {img_path} not found.")
                all_embeddings.append(np.zeros((1, 512)))
        
        self.embeddings = np.vstack(all_embeddings)
        print(f"Successfully indexed {len(self.embeddings)} images.")

    def search_by_text(self, query, top_k=5):
        text_emb = self.get_text_embedding(query)
        similarities = np.dot(self.embeddings, text_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def search_by_image(self, image_path, top_k=5):
        img_emb = self.get_image_embedding(image_path)
        similarities = np.dot(self.embeddings, img_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def save_index(self, path="data/index.npy"):
        np.save(path, self.embeddings)

    def load_index(self, index_path="data/index.npy", metadata_path="data/metadata.csv"):
        self.embeddings = np.load(index_path)
        self.metadata = pd.read_csv(metadata_path)
