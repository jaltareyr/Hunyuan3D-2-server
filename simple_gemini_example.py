#!/usr/bin/env python3
"""
Simple example of using the Gemini text-to-3D API.

This script demonstrates how to generate a 3D model from a text description
using the Hunyuan3D API with Gemini integration.
"""

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
    
    # Request payload
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "texture": False,
        "type": "glb"
    }
    
    print(f"Generating 3D model from text: '{text_prompt}'")
    print("This may take a minute...")
    
    # Make the request
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            # Save the model
            with open(output_filename, "wb") as f:
                f.write(response.content)
            print(f"✅ Success! Model saved to {output_filename}")
            print(f"💡 Note: The Gemini-generated image is saved on the server as *_output_img.png")
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
        "A cute cartoon robot with big eyes",
        "robot.glb"
    )
    
    # Example 2: More detailed description
    generate_3d_from_text(
        "A medieval sword with intricate engravings on the blade and a jeweled handle",
        "sword.glb"
    )
    
    # Example 3: Organic object
    generate_3d_from_text(
        "A red apple sitting on a wooden table",
        "apple.glb"
    )
