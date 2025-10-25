# Loading Local Models Guide

## Your Model Structure

You have your model located at:
```
/root/Hunyuan3D-2-server/models/Hunyuan3D-2.1/
  hunyuan3d-dit-v2-1/
    config.yaml
    model.fp16.ckpt
```

## How to Load It

### Option 1: Using the `--models_dir` and `--use_ckpt` flags (Recommended)

Start the API server with these parameters:

```bash
python api_server.py \
  --models_dir /root/Hunyuan3D-2-server/models \
  --model_path Hunyuan3D-2.1 \
  --model_subfolder hunyuan3d-dit-v2-1 \
  --use_ckpt \
  --port 8080 \
  --device cuda
```

**What this does:**
- `--models_dir`: Sets the base directory where models are stored
- `--model_path`: The model name (relative to models_dir)
- `--model_subfolder`: The subfolder containing config.yaml and model files
- `--use_ckpt`: Tells the system to use `.ckpt` files instead of `.safetensors`
- The system will look for: `{models_dir}/{model_path}/{model_subfolder}/model.fp16.ckpt`

### Option 2: Using Environment Variable

Set the environment variable before starting:

```bash
export HY3DGEN_MODELS=/root/Hunyuan3D-2-server/models

python api_server.py \
  --model_path Hunyuan3D-2.1 \
  --model_subfolder hunyuan3d-dit-v2-1 \
  --use_ckpt \
  --port 8080 \
  --device cuda
```

## Complete Command for Your Setup

Based on your error log, use this exact command:

```bash
python api_server.py \
  --models_dir /root/Hunyuan3D-2-server/models \
  --model_path Hunyuan3D-2.1 \
  --model_subfolder hunyuan3d-dit-v2-1 \
  --use_ckpt \
  --port 8080 \
  --device cuda
```

## New Command-Line Arguments

- `--use_ckpt`: Use `.ckpt` checkpoint files instead of `.safetensors` (required for your model)
- `--models_dir`: Base directory for models (sets `HY3DGEN_MODELS` environment variable)

## Verification

The server will log:
1. `Set HY3DGEN_MODELS to /root/Hunyuan3D-2-server/models`
2. `Try to load model from local path: /root/Hunyuan3D-2-server/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1`
3. `Loading model from /root/Hunyuan3D-2-server/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/model.fp16.ckpt`

## Troubleshooting

### If you still get "Model file not found":
1. Verify the file exists:
   ```bash
   ls -la /root/Hunyuan3D-2-server/models/Hunyuan3D-2.1/hunyuan3d-dit-v2-1/
   ```
   You should see:
   - `config.yaml`
   - `model.fp16.ckpt`

2. Check permissions:
   ```bash
   chmod -R 755 /root/Hunyuan3D-2-server/models/
   ```

### If it tries to download from HuggingFace:
This means the local path wasn't found. Double-check:
- The `--models_dir` path is correct
- The `--model_path` matches your directory name
- The `--model_subfolder` matches your subdirectory name

## Example with Texture Support

```bash
python api_server.py \
  --models_dir /root/Hunyuan3D-2-server/models \
  --model_path Hunyuan3D-2.1 \
  --model_subfolder hunyuan3d-dit-v2-1 \
  --use_ckpt \
  --enable_tex \
  --port 8080 \
  --device cuda
```
