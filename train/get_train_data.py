import os, zipfile
from huggingface_hub import snapshot_download, list_repo_files

REPO_ID = "seshing/openfacades-dataset"   # dataset repo
DEST    = "data"                           # local folder

def main():
    os.makedirs(DEST, exist_ok=True)
    files = list_repo_files(REPO_ID, repo_type="dataset")
    repo_dir = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DEST,
        allow_patterns=[
            "train.zip",          
            "img/train.zip",     
            "jsonl/train.jsonl",
        ],
        max_workers=4, 
        resume_download=True,
    )

    cand = [p for p in ("train.zip", "img/train.zip") if os.path.exists(os.path.join(repo_dir, p))]
    if not cand:
        raise FileNotFoundError("Could not find train.zip (checked repo root and img/)")
    zip_path = os.path.join(repo_dir, cand[0])

    # Extract into data/img
    os.makedirs(DEST, exist_ok=True)
    print(f"Extracting {zip_path} → {DEST}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST)

    print("Download + extract complete.")
    print(f"Images in: {DEST}")
    print(f"JSONL in:  {os.path.join(repo_dir, 'jsonl')}")

if __name__ == "__main__":
    main()
