import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import os

class XRaySearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Version marker for logs - we use v8.0.0 for the Pure Numpy fix
        print("--- X-Ray Search Engine Version: 8.0.0 (Pure Numpy Math Fix) ---")
        
        # Cloud environments often have unstable CUDA drivers for background tasks
        # We force CPU for extreme stability, especially with only 500 images
        self.device = "cpu"
        print(f"Loading search engine on {self.device}...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            # Final fallback
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
        self.embeddings = None
        self.metadata = None

    def _numpy_normalize(self, x):
        """Pure Numpy normalization. Globally stable across all environments."""
        # Ensure we have a numpy array
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        else:
            x = np.array(x)
            
        # Clean up any dimension weirdness
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        # Standard L2 Normalization in Numpy
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        # Prevent division by zero
        return x / (norm + 1e-12)

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        # Ensure return_tensors matches our CPU device
        inputs = self.processor(images=image, return_tensors="pt").to("cpu")
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # Convert to numpy and normalize
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
        # text_emb is already a normalized numpy array
        text_emb = self.get_text_embedding(query)
        # Similarity calculation in Pure Numpy
        similarities = np.dot(self.embeddings, text_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def search_by_image(self, image_path, top_k=5):
        # img_emb is already a normalized numpy array
        img_emb = self.get_image_embedding(image_path)
        # Similarity calculation in Pure Numpy
        similarities = np.dot(self.embeddings, img_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def save_index(self, path="data/index.npy"):
        np.save(path, self.embeddings)

    def load_index(self, index_path="data/index.npy", metadata_path="data/metadata.csv"):
        self.embeddings = np.load(index_path)
        self.metadata = pd.read_csv(metadata_path)
