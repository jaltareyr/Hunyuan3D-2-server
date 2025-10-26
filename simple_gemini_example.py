import base64
import os
from pathlib import Path
import requests

def generate_obj(
    server_url: str,
    out_path: str | None = None,
    *,
    image_path: str | None = None,
    text: str | None = None,
    seed: int = 1234,
    octree_resolution: int = 256,
    num_inference_steps: int = 40,
    guidance_scale: float = 5.0,
    face_count: int = 40000,
    timeout: int = 600,
) -> str:
    url = server_url.rstrip("/") + "/generate"

    if image_path:
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        with open(p, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        payload = {
            "image": img_b64,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "face_count": face_count,
        }
    else:
        if not text:
            raise ValueError("Provide either image_path or text.")
        payload = {
            "text": text,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "face_count": face_count,
        }

    resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    if resp.status_code != 200:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text[:500]
        raise RuntimeError(f"Server error ({resp.status_code}): {detail}")

    # Derive filename from Content-Disposition if present
    filename = "model.obj"
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        # Handles filename="abc.obj" or filename=abc.obj
        fn = cd.split("filename=")[-1].strip().strip('"').strip("'")
        if fn:
            filename = fn

    save_to = out_path or filename
    os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)

    with open(save_to, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    # Basic sanity check
    if not os.path.getsize(save_to):
        raise RuntimeError("Downloaded file is empty.")

    return save_to


# --- Example usage ---
if __name__ == "__main__":
    # 1) Text → OBJ
    path1 = generate_obj(
        "https://beckett-unaffiliated-unallegorically.ngrok-free.dev",
        out_path="outputs/model.obj",
        text="Big Tree (subject in center, plain background; diagonal angle for max coverage)"
    )
    print("Saved:", path1)
