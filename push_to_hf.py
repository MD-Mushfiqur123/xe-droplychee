#!/usr/bin/env python3
\"\"\"
push_to_hf.py - Securely upload m-droplychee model directory to Hugging Face Hub.
\"\"\"
import os
import sys
import argparse
from pathlib import Path

try:
    from huggingface_hub import HfApi, upload_folder, get_token
except ImportError:
    print(\"[ERROR] 'huggingface_hub' is required. Run: pip install huggingface_hub\", file=sys.stderr)
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description=\"Upload m-droplychee model to Hugging Face\")
    parser.add_argument(\"--repo-id\", \"-r\", type=str, required=True, help=\"Target Hugging Face repo (e.g. 'your-username/m-droplychee')\")
    parser.add_argument(\"--folder\", \"-f\", type=str, default=\"./hf_model\", help=\"Local folder containing model files\")
    parser.add_argument(\"--token\", \"-t\", type=str, default=None, help=\"Hugging Face write token\")
    parser.add_argument(\"--private\", action=\"store_true\", help=\"Make repository private\")
    args = parser.parse_args()

    token = args.token or os.getenv(\"HF_TOKEN\") or get_token()
    if not token:
        print(\"[ERROR] Hugging Face Token required. Set HF_TOKEN environment variable or pass --token\", file=sys.stderr)
        sys.exit(1)

    folder_path = Path(args.folder).resolve()
    if not folder_path.exists():
        print(f\"[ERROR] Folder {folder_path} does not exist!\", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    user_info = api.whoami(token=token)
    username = user_info.get(\"name\")
    print(f\"Authenticated as: {username}\")

    repo_id = args.repo_id
    if \"/\" not in repo_id:
        repo_id = f\"{username}/{repo_id}\"

    print(f\"Creating/Verifying repository: {repo_id} (private={args.private})...\")
    api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True, token=token)

    print(f\"Uploading folder '{folder_path}' to '{repo_id}'...\")
    upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        token=token,
        commit_message=\"Initial release of m-droplychee v4 (1/10th DeepSeek-V3/V4 Replica)\"
    )

    print(\"=\" * 60)
    print(\" SUCCESS! Model uploaded to Hugging Face:\")
    print(f\" https://huggingface.co/{repo_id}\")
    print(\"=\" * 60)

if __name__ == \"__main__\":
    main()
