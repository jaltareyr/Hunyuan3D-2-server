# Gemini Text-to-3D Integration

This document describes how to use the Gemini API integration for generating 3D models from text descriptions.

## Overview

The API server now supports generating 3D models directly from text descriptions by using Google's Gemini API to first generate an image, which is then processed by the Hunyuan3D pipeline to create a 3D model.

## Setup

1. Install the required dependencies:
```bash
pip install google-genai
```

2. Set up your Gemini API credentials. You'll need to have the `GOOGLE_API_KEY` environment variable set:
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Get your API key from: https://ai.google.dev/

**Note**: The Gemini API key is only required when using text-to-3D generation. If you're only using image-to-3D, you can skip this step. The Gemini client initializes lazily (only when first needed).

## Usage

### Using the `/generate` endpoint with text

You can now send a POST request to the `/generate` endpoint with a `text` parameter instead of an `image` parameter:

```python
import requests
import json

# Text-to-3D generation
response = requests.post(
    "http://localhost:8081/generate",
    json={
        "text": "A futuristic robot holding a glowing orb",
        "seed": 1234,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "texture": False,
        "type": "glb"
    }
)

# Save the generated 3D model
with open("output.glb", "wb") as f:
    f.write(response.content)
```

### Example with texture generation

```python
response = requests.post(
    "http://localhost:8081/generate",
    json={
        "text": "A detailed medieval castle on a hilltop",
        "seed": 5678,
        "octree_resolution": 256,
        "num_inference_steps": 10,
        "guidance_scale": 7.0,
        "texture": True,
        "face_count": 40000,
        "type": "glb"
    }
)

with open("textured_output.glb", "wb") as f:
    f.write(response.content)
```

### Using the async `/send` endpoint

For longer-running generations, you can use the async endpoint:

```python
# Send request
response = requests.post(
    "http://localhost:8081/send",
    json={
        "text": "A cartoon character with large eyes",
        "seed": 9999,
        "texture": True
    }
)

uid = response.json()["uid"]

# Check status
import time
while True:
    status_response = requests.get(f"http://localhost:8081/status/{uid}")
    status_data = status_response.json()
    
    if status_data["status"] == "completed":
        # Decode and save the model
        import base64
        model_data = base64.b64decode(status_data["model_base64"])
        with open(f"{uid}.glb", "wb") as f:
            f.write(model_data)
        break
    
    print("Still processing...")
    time.sleep(5)
```

## Parameters

- `text` (string, required when `image` is not provided): Text description of the 3D model to generate
- `image` (base64 string, optional): Base64-encoded image (alternative to text)
- `seed` (int, default: 1234): Random seed for reproducibility
- `octree_resolution` (int, default: 128): Resolution for octree-based mesh generation
- `num_inference_steps` (int, default: 5): Number of denoising steps
- `guidance_scale` (float, default: 5.0): Guidance scale for generation
- `texture` (bool, default: false): Whether to generate textures
- `face_count` (int, default: 40000): Maximum number of faces when texture is enabled
- `type` (string, default: "glb"): Output format (e.g., "glb", "obj")

## How It Works

1. When a `text` parameter is provided (and no `image`), the server uses the Gemini API to generate an image from the text description
2. The generated image is saved to the server as `{uid}_output_img.png` in the `gradio_cache` directory
3. The generated image undergoes background removal
4. The processed image is fed into the Hunyuan3D pipeline to generate the 3D mesh
5. Optionally, textures are applied if requested
6. The final 3D model is returned in the specified format

## Notes

- The Gemini API requires an active API key and internet connection
- Image generation may add a few seconds to the total processing time
- For best results, provide detailed and specific text descriptions
- The quality of the final 3D model depends on both the Gemini image generation and the Hunyuan3D processing
- The Gemini-generated image is automatically saved on the server as `{uid}_output_img.png` in the `gradio_cache` directory for reference
