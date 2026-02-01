import os
import pandas as pd

SAVE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"

def main():
    if not os.path.exists(SAVE_DIR):
        print("Error: images folder not found.")
        return
    
    files = os.listdir(SAVE_DIR)
    metadata = []
    
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Category is the first part of the filename before the underscore
            parts = f.split('_')
            category = parts[0] if len(parts) > 1 else "unknown"
            
            # Source depends on category mapping in our collector
            if category in ["chest", "skull", "dental", "spine", "fracture", "hand", "knee", "shoulder", "pelvis", "ankle", "elbow"]:
                source = "Open-I (NLM)"
            elif "gh" in f:
                source = "GitHub"
            elif "medpix" in f:
                source = "MedPix"
            else:
                source = "Wikimedia Commons"
                
            metadata.append({
                "image_name": f,
                "source_url": "N/A (Regenerated)",
                "category": category,
                "website": source
            })
            
    df = pd.DataFrame(metadata)
    df.to_csv(METADATA_FILE, index=False)
    print(f"Regenerated metadata for {len(df)} images.")

if __name__ == "__main__":
    main()
