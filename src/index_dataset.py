import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search_engine import XRaySearchEngine

def main():
    metadata_path = "data/metadata.csv"
    image_dir = "data/images"
    index_path = "data/index.npy"

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Run data_collector script first.")
        return

    # Check image count
    img_list = os.listdir(image_dir)
    print(f"Found {len(img_list)} images in {image_dir}.")
    
    if len(img_list) < 400:
        print("Warning: Image count is less than 500. You might want to wait for the downloader to finish.")

    engine = XRaySearchEngine()
    engine.index_dataset(metadata_path, image_dir)
    engine.save_index(index_path)
    print(f"Indexing complete. Saved to {index_path}")

if __name__ == "__main__":
    main()
