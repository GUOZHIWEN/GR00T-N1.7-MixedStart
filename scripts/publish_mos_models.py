#!/usr/bin/env python
"""Publish the MoS model folders with the official ModelScope Python SDK.

Uploads model weights, processor files, figures and LICENSE to the repository
root. The remote README.md is deliberately left untouched. No Git, download-back
or independent hash verification is used.
"""
import argparse
import os
from pathlib import Path


DEFAULT_REPO = "joyogzw/GR00T-N1.7-MixedStart"
UPLOAD_FOLDERS = ("mos_best", "mos_best_no_mem", "assets")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("release"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--cache-dir", type=Path,
                        default=Path(".cache/mos_model_publication"))
    args = parser.parse_args()

    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if not token:
        parser.error("Set MODELSCOPE_API_TOKEN before running this script")
    source = args.source_dir.resolve()
    for name in (*UPLOAD_FOLDERS, "LICENSE"):
        if not (source / name).exists():
            raise FileNotFoundError(f"missing release input: {source / name}")

    try:
        from modelscope.hub.api import HubApi
    except ImportError as exc:
        raise RuntimeError("Install the official SDK first: uv pip install -U modelscope") from exc

    api = HubApi(token=token)
    for name in UPLOAD_FOLDERS:
        print(f"Uploading {source / name} -> {args.repo_id}/{name}", flush=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(source / name),
            path_in_repo=name,
            repo_type="model",
            revision=args.revision,
            commit_message=f"Release {name}",
            commit_description="Official MoS model release; uploaded with ModelScope Python SDK.",
            tracker_path=str(args.cache_dir / name),
            token=token,
        )
    print(f"Uploading {source / 'LICENSE'} -> {args.repo_id}/LICENSE", flush=True)
    api.upload_file(
        repo_id=args.repo_id,
        path_or_fileobj=str(source / "LICENSE"),
        path_in_repo="LICENSE",
        repo_type="model",
        revision=args.revision,
        commit_message="Add Apache-2.0 license",
        token=token,
    )
    print(f"Upload finished: https://www.modelscope.cn/models/{args.repo_id}")
    print("The remote README.md was not uploaded or modified by this script.")


if __name__ == "__main__":
    main()
