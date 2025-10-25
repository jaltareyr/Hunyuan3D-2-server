# Gemini API Integration - Summary of Changes

## Overview
Added Gemini API integration to enable text-to-3D generation. Users can now provide text descriptions instead of images to generate 3D models. The Gemini API generates an image from the text, which is then processed through the existing Hunyuan3D pipeline.

## Files Modified

### 1. `api_server.py`
**Changes:**
- Added imports for Gemini API:
  ```python
  from google import genai
  from google.genai import types
  ```
  
- Added Gemini client initialization in `ModelWorker.__init__()`:
  ```python
  self.gemini_client = genai.Client()
  ```
  
- Added new method `generate_image_from_text_gemini()`:
  - Takes a text prompt as input
  - Calls Gemini API to generate an image
  - Returns PIL Image object
  - Logs the process for debugging
  
- Updated `generate()` method:
  - When `text` parameter is provided (instead of `image`), it calls `generate_image_from_text_gemini()`
  - The generated image is then processed through the existing pipeline (background removal, 3D generation, optional texturing)
  - No other changes to the existing flow

### 2. `requirements.txt`
**Changes:**
- Added `google-genai` to the dependencies under the "Demo only" section

## New Files Created

### 1. `GEMINI_TEXT_TO_3D.md`
Comprehensive documentation including:
- Setup instructions
- API credentials configuration
- Usage examples for both sync (`/generate`) and async (`/send`) endpoints
- Parameter descriptions
- How the integration works
- Notes and best practices

### 2. `test_gemini_text_to_3d.py`
Test script with three test modes:
- **Sync test**: Tests the `/generate` endpoint with text input
- **Async test**: Tests the `/send` + `/status` flow with text input
- **Textured test**: Tests text-to-3D with texture generation enabled
- Includes command-line arguments for customization
- Provides detailed console output with emojis for better UX

## API Usage

### Request Format
The `/generate` endpoint now accepts either:

**Option 1: Image input (existing)**
```json
{
  "image": "base64_encoded_image_string",
  "seed": 1234,
  ...
}
```

**Option 2: Text input (new)**
```json
{
  "text": "A futuristic robot holding a glowing orb",
  "seed": 1234,
  "octree_resolution": 128,
  "num_inference_steps": 5,
  "guidance_scale": 5.0,
  "texture": false,
  "type": "glb"
}
```

### Response
- Same as before: Returns the 3D model file (GLB format by default)
- For async endpoint: Returns UID for status checking

## Setup Requirements

1. Install the new dependency:
   ```bash
   pip install google-genai
   ```

2. Set up Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. Start the server as usual:
   ```bash
   python api_server.py --enable_tex
   ```

## How It Works

1. **User sends text prompt** → API receives text parameter
2. **Gemini generates image** → `generate_image_from_text_gemini()` calls Gemini API
3. **Image processing** → Background removal (existing rembg pipeline)
4. **3D generation** → Hunyuan3D pipeline (no changes to existing logic)
5. **Optional texturing** → If `texture: true`, applies textures
6. **Return model** → GLB file returned to user

## Key Features

✅ **Non-breaking changes**: Existing image-based API calls work exactly as before
✅ **Flexible input**: Support both image and text inputs
✅ **Seamless integration**: Gemini output flows directly into existing pipeline
✅ **Logging**: All steps are logged for debugging
✅ **Error handling**: Proper error messages if Gemini fails
✅ **Documentation**: Complete docs and test scripts provided

## Testing

Run the test script:
```bash
# Test sync endpoint
python test_gemini_text_to_3d.py --test sync --prompt "A cute robot"

# Test async endpoint
python test_gemini_text_to_3d.py --test async --prompt "A medieval castle"

# Test with textures
python test_gemini_text_to_3d.py --test textured

# Run all tests
python test_gemini_text_to_3d.py --test all
```

## Notes

- The Gemini API key must be set in environment variables
- Image generation adds a few seconds to total processing time
- More detailed prompts generally produce better results
- The `/send` async endpoint is recommended for longer-running generations
- Texture generation requires `--enable_tex` flag when starting the server
