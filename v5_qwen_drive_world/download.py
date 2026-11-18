"""Pinned official artifacts only; Hugging Face owns resumable transfers."""
from .runtime import ROOT, configure
configure()
import hashlib
import json
import traceback
from huggingface_hub import HfApi, snapshot_download

REVISION = "28484089a7cc8c335cf5089fb0745cf7c49b6eaa"
DEST = ROOT / "models/v5-qwen-drive-world/Qwen-Drive-1.0-4B"
REPORT = ROOT / "artifacts/v5-qwen-drive-world/download.json"


def main():
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    state = {"repository": "Qwen/Qwen-Drive-1.0-4B", "revision": REVISION, "status": "downloading", "destination": str(DEST)}
    REPORT.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        snapshot_download(state["repository"], revision=REVISION, local_dir=DEST,
                          allow_patterns=["*.json", "*.jinja", "*.txt", "model.safetensors", "perception/*", "planner-rl/*"],
                          ignore_patterns=["planner-sft/*"], max_workers=3)
        info = HfApi().model_info(state["repository"], revision=REVISION, files_metadata=True)
        checked = []
        for entry in info.siblings:
            if entry.rfilename not in ["model.safetensors", "planner-rl/model.safetensors", "perception/model.safetensors"]:
                continue
            path = DEST / entry.rfilename
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != entry.lfs.sha256 or path.stat().st_size != entry.size:
                raise ValueError(f"Official checksum mismatch: {entry.rfilename}")
            checked.append({"file": entry.rfilename, "sha256": digest, "bytes": entry.size})
        if len(checked) != 3:
            raise ValueError("Three required components not verified")
        state.update(status="verified", files=checked)
    except Exception:
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        REPORT.write_text(json.dumps(state, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
