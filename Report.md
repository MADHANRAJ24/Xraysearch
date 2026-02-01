# X-Ray Image Search Interface - Technical Report

## 1. Project Objective
The objective of this project is to build a prototype for an X-ray image search engine that allows users to find relevant medical images using either text-based queries (natural language) or image-based queries (reverse image search).

## 2. Dataset Collection
We collected a dataset of **500+ X-ray images** across multiple categories.
- **Total Images**: 500+ (Targeting 1000 for robustness)
- **Categories**: Chest, Dental, Spine, Fracture, Hand, Pelvis, Knee, Skull, Shoulder.
- **Sources**:
  1. **Open-I (NLM)**: A large-scale medical image repository.
  2. **GitHub (Dental-Xray-Dataset)**: A public dataset of dental radiographs.
  3. **Wikimedia Commons**: Public domain medical imaging.
  4. **MedPix (NIH)**: Database of medical images and case studies.
  5. **Mendeley Data**: Research datasets for bone fractures.

Each image is tracked in `data/metadata.csv` with its filename, original source URL, and category.

## 3. Search Engine Logic
The core search functionality is powered by **OpenAI's CLIP (Contrastive Language-Image Pre-training)** model.
- **Unified Embedding Space**: CLIP maps both text and images into the same 512-dimensional vector space.
- **Text-to-Image Search**: When a user enters a query like "chest pneumonia", the text is converted into a vector. We then find the image vectors with the highest **cosine similarity**.
- **Image-to-Image Search**: When a user uploads an image, it is converted into a vector, and we find the most similar images in the dataset by comparing their vector distances.
- **Efficiency**: Image embeddings are pre-computed (indexed) and stored as a NumPy array for near-instant search results.

## 4. User Interface
The interface is built using **Streamlit**, providing a modern and responsive web experience.
- **Search Tabs**: Separate tabs for text and image search.
- **Results Display**: High-resolution thumbnails displayed in a grid with category labels and similarity scores.
- **Scalability**: The UI dynamically updates based on the indexed dataset.

## 5. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download data (if not present): `python src/data_collector.py`
3. Index the dataset: `python src/index_dataset.py`
4. Start the app: `streamlit run app/main.py`
