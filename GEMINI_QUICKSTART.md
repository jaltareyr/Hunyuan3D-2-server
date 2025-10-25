# 🎨 Gemini Text-to-3D Quick Start

## What's New?

You can now generate 3D models directly from text descriptions! The API uses Google's Gemini to create images from your text, then generates 3D models using Hunyuan3D.

## Quick Setup

1. **Install the Gemini SDK:**
   ```bash
   pip install google-genai
   ```

2. **Set your Gemini API key (only needed for text-to-3D):**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```
   
   Get your API key from: https://ai.google.dev/
   
   **Note**: The API key is only required when using text input. Image-to-3D works without it.

3. **Start the server:**
   ```bash
   python api_server.py --enable_tex
   ```

## Simple Example

```python
import requests

response = requests.post(
    "http://localhost:8081/generate",
    json={
        "text": "A cute robot holding a glowing orb",
        "seed": 1234
    }
)

with open("robot.glb", "wb") as f:
    f.write(response.content)
```

## Try It Now

We've included example scripts:

```bash
# Simple usage example
python simple_gemini_example.py

# Full test suite
python test_gemini_text_to_3d.py --test all
```

## API Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | - | Text description (alternative to `image`) |
| `seed` | int | 1234 | Random seed for reproducibility |
| `octree_resolution` | int | 128 | Mesh resolution |
| `num_inference_steps` | int | 5 | Generation quality/speed |
| `guidance_scale` | float | 5.0 | How closely to follow the prompt |
| `texture` | bool | false | Generate textures (requires `--enable_tex`) |
| `type` | string | "glb" | Output format |

## How It Works

```
Text Prompt → Gemini Image → Save Image (output_img.png) → Background Removal → 3D Generation → Your Model!
```

## More Examples

See the documentation:
- 📖 **Full Documentation**: [GEMINI_TEXT_TO_3D.md](GEMINI_TEXT_TO_3D.md)
- 📋 **Changes Summary**: [CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)
- 🧪 **Test Script**: [test_gemini_text_to_3d.py](test_gemini_text_to_3d.py)

## Tips for Best Results

✅ Be specific and descriptive
✅ Mention materials, colors, and style
✅ Use `texture: true` for realistic models (slower)
✅ Increase `num_inference_steps` for higher quality

Example prompts:
- "A red ceramic vase with gold trim"
- "A wooden treasure chest with metal hinges"
- "A futuristic sci-fi helmet with blue lights"

---

**Note**: The existing image-based API still works exactly as before. Text input is an additional option!
