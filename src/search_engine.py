import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import os

class XRaySearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        # Version marker for logs - we use v7.0.0 for the ultra-defensive fix
        print("--- X-Ray Search Engine Version: 7.0.0 (Hyper-Defensive Fix) ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading search engine on {self.device}...")
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.device = "cpu"
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            
        self.embeddings = None
        self.metadata = None

    def _safe_normalize(self, features):
        """Ultra-robust normalization that handles various types and environment quirks."""
        # Ensure we are working with a torch tensor
        if not torch.is_tensor(features):
            print(f"Warning: Features is not a tensor, type is {type(features)}")
            try:
                features = torch.tensor(np.array(features)).to(self.device)
            except:
                features = torch.as_tensor(features).to(self.device)

        # Ensure features are floats
        features = features.float()

        # Manual normalization to avoid F.normalize or .norm() issues on Cloud
        # We use explicit multiplication and sum to avoid any high-level function issues
        sq_sum = (features * features).sum(dim=-1, keepdim=True)
        # Add a tiny epsilon and take sqrt to get the norm
        norm = torch.sqrt(sq_sum + 1e-12)
        return features / norm

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_image_features(**inputs)
            
        # CLIP model can sometimes return objects or tensors depending on version
        # We ensure it's a tensor before normalizing
        return self._safe_normalize(outputs)

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            
        return self._safe_normalize(outputs)

    def index_dataset(self, metadata_path, image_dir):
        print("Indexing dataset...")
        self.metadata = pd.read_csv(metadata_path)
        all_embeddings = []
        
        for idx, row in self.metadata.iterrows():
            img_path = os.path.join(image_dir, row['image_name'])
            if os.path.exists(img_path):
                emb = self.get_image_embedding(img_path)
                all_embeddings.append(emb.cpu().numpy())
            else:
                print(f"Warning: Image {img_path} not found.")
                all_embeddings.append(np.zeros((1, 512)))
        
        self.embeddings = np.vstack(all_embeddings)
        print(f"Successfully indexed {len(self.embeddings)} images.")

    def search_by_text(self, query, top_k=5):
        text_emb = self.get_text_embedding(query).cpu().numpy()
        similarities = np.dot(self.embeddings, text_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def search_by_image(self, image_path, top_k=5):
        img_emb = self.get_image_embedding(image_path).cpu().numpy()
        similarities = np.dot(self.embeddings, img_emb.T).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.metadata.iloc[top_indices], similarities[top_indices]

    def save_index(self, path="data/index.npy"):
        np.save(path, self.embeddings)

    def load_index(self, index_path="data/index.npy", metadata_path="data/metadata.csv"):
        self.embeddings = np.load(index_path)
        self.metadata = pd.read_csv(metadata_path)
