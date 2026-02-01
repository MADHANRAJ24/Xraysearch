import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.search_engine import XRaySearchEngine

def main():
    engine = XRaySearchEngine()
    engine.load_index("data/index.npy", "data/metadata.csv")
    
    # Test 1: Text Search
    print("\nTesting Text Search for 'chest'...")
    results, scores = engine.search_by_text("chest x-ray", top_k=5)
    for i, (idx, row) in enumerate(results.iterrows()):
        print(f"{i+1}. {row['image_name']} | Category: {row['category']} | Score: {scores[i]:.4f}")

    # Test 2: Text Search for 'fracture'
    print("\nTesting Text Search for 'fracture'...")
    results, scores = engine.search_by_text("bone fracture", top_k=5)
    for i, (idx, row) in enumerate(results.iterrows()):
        print(f"{i+1}. {row['image_name']} | Category: {row['category']} | Score: {scores[i]:.4f}")

    # Test 3: Image Search (using first image as query)
    sample_img = os.path.join("data/images", engine.metadata.iloc[0]['image_name'])
    print(f"\nTesting Image Search for query: {sample_img}...")
    results, scores = engine.search_by_image(sample_img, top_k=5)
    for i, (idx, row) in enumerate(results.iterrows()):
        print(f"{i+1}. {row['image_name']} | Category: {row['category']} | Score: {scores[i]:.4f}")

if __name__ == "__main__":
    main()
