# generate_server.py  (updated)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: Returns ONLY a single .obj file; no GLB, no ZIP, no pickle.
# Texture/painter code lives in hy3dgen/texpaint_util.py
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import asyncio
import base64
import logging
import logging.handlers
import os
import sys
import tempfile
import threading
import traceback
import uuid
from io import BytesIO

import torch
import trimesh
import uvicorn
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from google import genai
from google.genai import types

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FloaterRemover, DegenerateFaceRemover, FaceReducer
# texture utility (new)
from hy3dgen.texpaint_util import colorize_mesh_from_reference

LOGDIR = '.'
SAVE_DIR = 'gradio_cache'
os.makedirs(SAVE_DIR, exist_ok=True)

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."
handler = None
worker_id = str(uuid.uuid4())[:6]


def build_logger(logger_name, logger_filename):
    global handler
    formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S",)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    stdout_logger = logging.getLogger("stdout"); stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO); sys.stdout = sl

    stderr_logger = logging.getLogger("stderr"); stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR); sys.stderr = sl

    logger = logging.getLogger(logger_name); logger.setLevel(logging.INFO)
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler_new = logging.handlers.TimedRotatingFileHandler(filename, when='D', utc=True, encoding='UTF-8')
        handler_new.setFormatter(formatter); handler = handler_new
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)
    return logger


class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)
    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line
    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


logger = build_logger("controller", f"{SAVE_DIR}/controller.log")


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


class ModelWorker:
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2mini',
                 model_subfolder='hunyuan3d-dit-v2-mini-turbo',
                 device='cuda',
                 use_safetensors=True):
        self.model_path = model_path
        self.model_subfolder = model_subfolder
        self.worker_id = worker_id
        self.device = device
        logger.info(f"Loading the model {model_path} on worker {worker_id} ...")
        if model_subfolder:
            logger.info(f"Using subfolder {model_subfolder} for model {model_path}")

        self.rembg = BackgroundRemover()
        self.gemini_client = None

        pipeline_kwargs = dict(use_safetensors=use_safetensors, device=device)
        if model_subfolder:
            pipeline_kwargs["subfolder"] = model_subfolder
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path, **pipeline_kwargs)
        self.pipeline.enable_flashvdm(mc_algo='mc')

    def generate_image_from_text_gemini(self, text_prompt):
        logger.info(f"Generating image from text using Gemini: {text_prompt}")
        if self.gemini_client is None:
            try:
                self.gemini_client = genai.Client()
                logger.info("Initialized Gemini client")
            except ValueError as e:
                error_msg = ("Failed to initialize Gemini client. Please set GOOGLE_API_KEY environment variable. "
                             "Get your API key from: https://ai.google.dev/")
                logger.error(error_msg); raise ValueError(error_msg) from e

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[text_prompt],
        )

        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                logger.info("Successfully generated image from text using Gemini")
                return image
            elif getattr(part, "text", None) is not None:
                logger.info(f"Gemini response text: {part.text}")
        raise ValueError("Gemini did not return an image")

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {"speed": 1, "queue_length": self.get_queue_length()}

    @torch.inference_mode()
    def generate(self, uid, params):
        # 1) Acquire input image (base64 or Gemini-from-text)
        if 'image' in params:
            image = load_image_from_base64(params["image"])
        else:
            if 'text' not in params:
                raise ValueError("No input image or text provided")
            prompt = (
                f"Generate a high-quality image of {params['text']}. "
                "Note that the request object should be at the center of the image with a clear plain background. "
                "This image will be used for 3D model generation so provide the image as clear as possible."
            )
            image = self.generate_image_from_text_gemini(prompt)

        # 2) Save a copy of the (pre-rembg) reference image for palette extraction
        ref_img_path = os.path.join(SAVE_DIR, f'{str(uid)}_ref.png')
        image.save(ref_img_path)

        # 3) Background removal for shape gen
        image_no_bg = self.rembg(image)
        params['image'] = image_no_bg

        # 4) Generate mesh
        seed = params.get("seed", 1234)
        params['generator'] = torch.Generator(self.device).manual_seed(seed)
        params['octree_resolution'] = params.get("octree_resolution", 128)
        params['num_inference_steps'] = params.get("num_inference_steps", 5)
        params['guidance_scale'] = params.get('guidance_scale', 5.0)
        params['mc_algo'] = 'mc'

        import time
        start_time = time.time()
        mesh = self.pipeline(**params)[0]
        logger.info("--- %s seconds ---" % (time.time() - start_time))

        # 5) Light cleanup before painting
        mesh = FloaterRemover()(mesh)
        mesh = DegenerateFaceRemover()(mesh)
        mesh = FaceReducer()(mesh, max_facenum=params.get('face_count', 40000))

        # 6) Paint vertex colors using the saved reference image
        mesh = colorize_mesh_from_reference(
            mesh,
            reference_image_path=ref_img_path,
            seed=seed,
            octaves=4,
            base_freq=0.8,
            smoothing_iters=24,
            world_scale=1.0,
            center_bias_uv=0.15,
            local_weight=0.60,
            palette_snap=0.35
        )

        # 7) Export ONLY OBJ
        obj_path = os.path.join(SAVE_DIR, f'{str(uid)}.obj')
        mesh.export(obj_path)

        # free VRAM
        torch.cuda.empty_cache()
        return obj_path, os.path.basename(obj_path), uid


# ─────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://scenergy.design"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate")
async def generate(request: Request):
    logger.info("Worker generating...")
    params = await request.json()
    uid = uuid.uuid4()
    try:
        file_path, download_filename, uid = worker.generate(uid, params)
        # Return ONLY the OBJ file
        return FileResponse(file_path, filename=download_filename, media_type="text/plain")
    except ValueError as e:
        traceback.print_exc()
        print("Caught ValueError:", e)
        return JSONResponse({"text": server_error_msg, "error_code": 1}, status_code=404)
    except torch.cuda.CudaError as e:
        print("Caught torch.cuda.CudaError:", e)
        return JSONResponse({"text": server_error_msg, "error_code": 1}, status_code=404)
    except Exception as e:
        print("Caught Unknown Error", e)
        traceback.print_exc()
        return JSONResponse({"text": server_error_msg, "error_code": 1}, status_code=404)


@app.post("/send")
async def send(request: Request):
    logger.info("Worker send...")
    params = await request.json()
    uid = uuid.uuid4()
    threading.Thread(target=worker.generate, args=(uid, params,)).start()
    return JSONResponse({"uid": str(uid)}, status_code=200)


@app.get("/status/{uid}")
async def status(uid: str):
    obj_path = os.path.join(SAVE_DIR, f'{uid}.obj')
    exists = os.path.exists(obj_path)
    print(obj_path, exists)
    if not exists:
        return JSONResponse({'status': 'processing'}, status_code=200)
    else:
        # For OBJ, just report completion (no base64 for large text files)
        return JSONResponse({'status': 'completed'}, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2mini')
    parser.add_argument("--model_subfolder", type=str, default='hunyuan3d-dit-v2-mini-turbo')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument('--use_ckpt', action='store_true', help='Use .ckpt files instead of .safetensors')
    parser.add_argument('--models_dir', type=str, default=None, help='Directory containing models (sets HY3DGEN_MODELS)')
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.models_dir:
        os.environ['HY3DGEN_MODELS'] = args.models_dir
        logger.info(f"Set HY3DGEN_MODELS to {args.models_dir}")

    model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    model_subfolder = args.model_subfolder if args.model_subfolder else None
    use_safetensors = not args.use_ckpt

    worker = ModelWorker(model_path=args.model_path,
                         model_subfolder=model_subfolder,
                         device=args.device,
                         use_safetensors=use_safetensors)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
