import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F

class XRaySearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading search engine on {self.device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.embeddings = None
        self.metadata = None

    def get_image_embedding(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
        
        # Ensure it's a tensor and normalize
        if not isinstance(image_features, torch.Tensor):
            image_features = torch.tensor(image_features)
        
        # Using manual normalization to avoid F.normalize issues in some environments
        norm = image_features.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return image_features / norm

    def get_text_embedding(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
        # Ensure it's a tensor and normalize
        if not isinstance(text_features, torch.Tensor):
            text_features = torch.tensor(text_features)
            
        norm = text_features.pow(2).sum(dim=-1, keepdim=True).sqrt()
        return text_features / norm

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
                # Add a zero embedding to maintain alignment
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
