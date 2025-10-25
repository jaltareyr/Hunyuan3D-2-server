#!/usr/bin/env python3
"""
Simple example of using the Gemini text-to-3D API.

This script demonstrates how to generate a 3D model from a text description
using the Hunyuan3D API with Gemini integration.
"""

import io
import os
import zipfile

import requests


def generate_3d_from_text(text_prompt, output_filename="model.glb"):
    """
    Generate a 3D model from a text description.
    
    Args:
        text_prompt: Text description of the object to generate
        output_filename: Name of the output file (default: model.glb)
    
    Returns:
        True if successful, False otherwise
    """
    # API endpoint
    api_url = "https://6b9eca0d4657.ngrok-free.app/generate"
    
    target_glb_path = os.path.abspath(output_filename)
    output_dir = os.path.dirname(target_glb_path)
    os.makedirs(output_dir, exist_ok=True)

    # Request payload
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 256,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "texture": False,
        "type": "glb",
        "return_gemini_image": True,
    }
    
    print(f"Generating 3D model from text: '{text_prompt}'")
    print("This may take a minute...")
    
    # Make the request
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")

            if "application/zip" in content_type:
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    zf.extractall(output_dir)
                    glb_member = next((name for name in zf.namelist() if name.endswith(".glb")), None)
                    png_member = next((name for name in zf.namelist() if name.endswith(".png")), None)

                if glb_member:
                    extracted_glb_path = os.path.join(output_dir, glb_member)
                    os.replace(extracted_glb_path, target_glb_path)
                    print(f"✅ Success! Model saved to {target_glb_path}")
                else:
                    print("⚠️ Warning: Model file missing from archive")

                if png_member:
                    image_path = os.path.join(output_dir, png_member)
                    print(f"🖼️ Gemini image saved to {image_path}")
                else:
                    print("⚠️ Warning: Gemini image missing from archive")
            else:
                with open(target_glb_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Success! Model saved to {target_glb_path}")
                print("ℹ️ Gemini image not requested; only the GLB model was returned.")
            return True
        else:
            print(f"❌ Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


# Example usage
if __name__ == "__main__":
    # Example 1: Simple object
    generate_3d_from_text(
        "A group of 10 people standing (Note: Always draw the object showing diagonal view to provide maximum details for a  3d model's use)",
        "output.glb"
    )
