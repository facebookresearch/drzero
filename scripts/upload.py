import os
from huggingface_hub import upload_file

repo_id = "PeterJinGo/wiki-18-e5-index"
path = "/home/peterjin/mnt/index/wiki-18"
for file in ["part_aa", "part_ab"]:
    upload_file(
        path_or_fileobj=os.path.join(path, file),  # File path
        path_in_repo=file,  # Destination filename in the repo
        repo_id=repo_id,  # Your dataset repo ID
        repo_type="dataset"
    )
