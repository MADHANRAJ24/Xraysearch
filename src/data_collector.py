import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

SAVE_DIR = "data/images"
METADATA_FILE = "data/metadata.csv"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR, exist_ok=True)

def download_image(args):
    url, filename, category, source = args
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            with open(os.path.join(SAVE_DIR, filename), "wb") as f:
                f.write(res.content)
            return {"image_name": filename, "source_url": url, "category": category, "website": source}
    except: pass
    return None

def fetch_openi(query, limit=100):
    url = f"https://openi.nlm.nih.gov/api/search?it=xg&m=1&n={limit}&query={query}"
    try:
        items = requests.get(url, timeout=10).json().get("list", [])
        return [("https://openi.nlm.nih.gov"+item["imgLarge"], f"{query}_{i}.png", query, "Open-I") for i, item in enumerate(items) if "imgLarge" in item]
    except: return []

def main():
    urls = []
    queries = ["chest", "skull", "dental", "spine", "fracture", "hand", "knee", "shoulder", "pelvis", "ankle", "elbow"]
    for q in queries:
        urls += fetch_openi(q, 100)
    
    # Backup Source: GitHub Dental
    urls += [(f"https://raw.githubusercontent.com/shivasankark/dental-xray-dataset/master/images/{i}.jpg", f"dental_gh_{i}.jpg", "dental", "GitHub") for i in range(1, 101)]

    print(f"Total target URLs: {len(urls)}")
    with ThreadPoolExecutor(max_workers=15) as ex:
        results = list(tqdm(ex.map(download_image, urls), total=len(urls)))
    
    df = pd.DataFrame([r for r in results if r])
    df.to_csv(METADATA_FILE, index=False)
    print(f"Grand total: {len(df)}")

if __name__ == "__main__": main()
