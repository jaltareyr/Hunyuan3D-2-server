#!/usr/bin/env python3
"""
Example script to test the Gemini text-to-3D API integration.

Usage:
    python test_gemini_text_to_3d.py
"""

import requests
import base64
import time
import argparse


def test_sync_text_to_3d(base_url, text_prompt, output_file="output.glb"):
    """Test synchronous text-to-3D generation using the /generate endpoint."""
    print(f"🚀 Testing synchronous text-to-3D generation...")
    print(f"📝 Prompt: {text_prompt}")
    
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "texture": False,
        "type": "glb"
    }
    
    print(f"⏳ Sending request to {base_url}/generate...")
    response = requests.post(f"{base_url}/generate", json=payload)
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Success! Model saved to {output_file}")
        print(f"💡 Note: The Gemini-generated image is saved on the server as *_output_img.png")
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def test_async_text_to_3d(base_url, text_prompt, output_file="async_output.glb"):
    """Test asynchronous text-to-3D generation using the /send and /status endpoints."""
    print(f"\n🚀 Testing asynchronous text-to-3D generation...")
    print(f"📝 Prompt: {text_prompt}")
    
    payload = {
        "text": text_prompt,
        "seed": 5678,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "texture": False,
        "type": "glb"
    }
    
    print(f"⏳ Sending async request to {base_url}/send...")
    response = requests.post(f"{base_url}/send", json=payload)
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False
    
    uid = response.json()["uid"]
    print(f"📋 Job UID: {uid}")
    
    # Poll for completion
    max_attempts = 60  # 5 minutes max
    attempt = 0
    
    while attempt < max_attempts:
        print(f"⏳ Checking status (attempt {attempt + 1}/{max_attempts})...")
        status_response = requests.get(f"{base_url}/status/{uid}")
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # Decode and save the model
                model_data = base64.b64decode(status_data["model_base64"])
                with open(output_file, "wb") as f:
                    f.write(model_data)
                print(f"✅ Success! Model saved to {output_file}")
                return True
            elif status_data["status"] == "processing":
                print("   Still processing...")
                time.sleep(5)
                attempt += 1
            else:
                print(f"❌ Unknown status: {status_data['status']}")
                return False
        else:
            print(f"❌ Error checking status: {status_response.status_code}")
            return False
    
    print(f"❌ Timeout: Model generation took too long")
    return False


def test_textured_generation(base_url, text_prompt, output_file="textured_output.glb"):
    """Test text-to-3D generation with textures."""
    print(f"\n🚀 Testing textured text-to-3D generation...")
    print(f"📝 Prompt: {text_prompt}")
    
    payload = {
        "text": text_prompt,
        "seed": 42,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "texture": False,
        "face_count": 40000,
        "type": "glb"
    }
    
    print(f"⏳ Sending request to {base_url}/generate...")
    print("⚠️  Note: This may take longer due to texture generation...")
    
    response = requests.post(f"{base_url}/generate", json=payload)
    
    if response.status_code == 200:
        with open(output_file, "wb") as f:
            f.write(response.content)
        print(f"✅ Success! Textured model saved to {output_file}")
        print(f"💡 Note: The Gemini-generated image is saved on the server as *_output_img.png")
        return True
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Gemini text-to-3D API integration")
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://6b9eca0d4657.ngrok-free.app",
        help="Base URL of the API server (default: http://localhost:8081)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A cute robot holding a glowing orb",
        help="Text prompt for 3D generation"
    )
    parser.add_argument(
        "--test",
        type=str,
        choices=["sync", "async", "textured", "all"],
        default="sync",
        help="Which test to run (default: sync)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎨 Gemini Text-to-3D API Test Script")
    print("=" * 60)
    
    if args.test == "sync" or args.test == "all":
        test_sync_text_to_3d(args.base_url, args.prompt)
    
    if args.test == "async" or args.test == "all":
        test_async_text_to_3d(args.base_url, args.prompt)
    
    if args.test == "textured" or args.test == "all":
        test_textured_generation(
            args.base_url,
            "A medieval castle on a hilltop with detailed stonework"
        )
    
    print("\n" + "=" * 60)
    print("🎉 Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
