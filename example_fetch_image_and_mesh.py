#!/usr/bin/env python3
"""
Example: Fetching both Gemini image and mesh from the API

This script demonstrates how to:
1. Request both the Gemini-generated image and the 3D mesh
2. Extract both files from the ZIP response
3. Load and display/process both the image and mesh
"""

import io
import os
import pickle
import zipfile

import requests
from PIL import Image


def generate_with_image_and_mesh(text_prompt, output_dir="output", return_pickle=False):
    """
    Generate a 3D mesh and get the Gemini-generated image.
    
    Args:
        text_prompt: Text description of the object
        output_dir: Directory to save outputs
        return_pickle: If True, request pickled mesh instead of GLB
    
    Returns:
        Tuple of (mesh_path, image_path) if successful, (None, None) otherwise
    """
    api_url = "http://localhost:8080/generate"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Request payload
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 256,
        "num_inference_steps": 40,
        "guidance_scale": 4.0,
        "texture": False,
        "return_gemini_image": True,  # Request Gemini image
        "return_pickle": return_pickle  # Request pickle or GLB
    }
    
    print(f"🎨 Generating from text: '{text_prompt}'")
    print(f"   Requesting: {'Pickled mesh' if return_pickle else 'GLB file'} + Gemini image")
    print("   This may take a minute...")
    
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "")
            
            if "application/zip" in content_type or "application/x-zip-compressed" in content_type:
                print("📦 Received ZIP file, extracting...")
                
                # Extract ZIP contents
                with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                    # List all files in the ZIP
                    print(f"   Files in archive: {', '.join(zf.namelist())}")
                    
                    # Extract all files
                    zf.extractall(output_dir)
                    
                    # Find the mesh file (either .pkl or .glb)
                    mesh_file = None
                    for name in zf.namelist():
                        if name.endswith(".pkl") or name.endswith(".glb"):
                            mesh_file = name
                            break
                    
                    # Find the image file
                    image_file = None
                    for name in zf.namelist():
                        if name.endswith(".png") or name.endswith(".jpg"):
                            image_file = name
                            break
                
                if mesh_file:
                    mesh_path = os.path.join(output_dir, mesh_file)
                    print(f"✅ Mesh saved: {mesh_path}")
                else:
                    print("⚠️  Warning: Mesh file not found in archive")
                    mesh_path = None
                
                if image_file:
                    image_path = os.path.join(output_dir, image_file)
                    print(f"🖼️  Image saved: {image_path}")
                else:
                    print("⚠️  Warning: Image file not found in archive")
                    image_path = None
                
                return mesh_path, image_path
            else:
                print("⚠️  Received single file (not a ZIP), only mesh available")
                mesh_ext = ".pkl" if return_pickle else ".glb"
                mesh_path = os.path.join(output_dir, f"model{mesh_ext}")
                with open(mesh_path, "wb") as f:
                    f.write(response.content)
                print(f"✅ Mesh saved: {mesh_path}")
                return mesh_path, None
        else:
            print(f"❌ Error: Server returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def process_image(image_path):
    """
    Load and display information about the Gemini-generated image.
    
    Args:
        image_path: Path to the image file
    """
    if not image_path or not os.path.exists(image_path):
        print("⚠️  No image to process")
        return None
    
    print(f"\n🖼️  Processing image: {image_path}")
    
    # Load the image
    img = Image.open(image_path)
    
    # Display image information
    print(f"   Format: {img.format}")
    print(f"   Size: {img.size[0]}x{img.size[1]} pixels")
    print(f"   Mode: {img.mode}")
    
    # You can perform additional operations
    # For example, create a thumbnail
    thumbnail_path = image_path.replace(".png", "_thumbnail.png")
    img.thumbnail((256, 256))
    img.save(thumbnail_path)
    print(f"   Thumbnail saved: {thumbnail_path}")
    
    return img


def process_mesh(mesh_path, return_pickle=False):
    """
    Load and display information about the mesh.
    
    Args:
        mesh_path: Path to the mesh file
        return_pickle: If True, load as pickle, else use trimesh
    """
    if not mesh_path or not os.path.exists(mesh_path):
        print("⚠️  No mesh to process")
        return None
    
    print(f"\n📦 Processing mesh: {mesh_path}")
    
    try:
        import trimesh
        
        if return_pickle:
            # Load pickled mesh
            with open(mesh_path, 'rb') as f:
                mesh = pickle.load(f)
            print(f"   Loaded from pickle")
        else:
            # Load GLB file
            mesh = trimesh.load(mesh_path)
            print(f"   Loaded GLB file")
        
        # Display mesh information
        print(f"   Vertices: {len(mesh.vertices):,}")
        print(f"   Faces: {len(mesh.faces):,}")
        print(f"   Volume: {mesh.volume:.2f}")
        print(f"   Surface area: {mesh.area:.2f}")
        print(f"   Bounds: {mesh.bounds.tolist()}")
        print(f"   Center: {mesh.centroid.tolist()}")
        
        # Export to different formats
        print(f"\n💾 Exporting to different formats...")
        base_name = os.path.splitext(mesh_path)[0]
        
        formats = ['obj', 'stl', 'ply']
        for fmt in formats:
            try:
                export_path = f"{base_name}.{fmt}"
                mesh.export(export_path)
                print(f"   ✅ Exported: {export_path}")
            except Exception as e:
                print(f"   ❌ Failed to export {fmt}: {e}")
        
        return mesh
        
    except Exception as e:
        print(f"❌ Error processing mesh: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_combined_visualization(image_path, mesh_path):
    """
    Create a simple visualization showing both the image and mesh info.
    
    Args:
        image_path: Path to the Gemini image
        mesh_path: Path to the mesh file
    """
    print(f"\n📊 Creating combined visualization...")
    
    try:
        import trimesh
        from PIL import Image, ImageDraw, ImageFont
        
        # Load image
        img = Image.open(image_path)
        
        # Load mesh
        if mesh_path.endswith('.pkl'):
            with open(mesh_path, 'rb') as f:
                mesh = pickle.load(f)
        else:
            mesh = trimesh.load(mesh_path)
        
        # Create a new image with both
        width = img.size[0] + 400
        height = max(img.size[1], 400)
        combined = Image.new('RGB', (width, height), 'white')
        
        # Paste the Gemini image
        combined.paste(img, (0, 0))
        
        # Add mesh info as text
        draw = ImageDraw.Draw(combined)
        x_offset = img.size[0] + 20
        y_offset = 20
        
        info_text = [
            "MESH INFORMATION",
            "",
            f"Vertices: {len(mesh.vertices):,}",
            f"Faces: {len(mesh.faces):,}",
            f"Volume: {mesh.volume:.2f}",
            f"Surface area: {mesh.area:.2f}",
            "",
            "Bounds:",
            f"  Min: [{mesh.bounds[0][0]:.2f}, {mesh.bounds[0][1]:.2f}, {mesh.bounds[0][2]:.2f}]",
            f"  Max: [{mesh.bounds[1][0]:.2f}, {mesh.bounds[1][1]:.2f}, {mesh.bounds[1][2]:.2f}]",
        ]
        
        for i, line in enumerate(info_text):
            draw.text((x_offset, y_offset + i * 25), line, fill='black')
        
        # Save combined visualization
        output_path = os.path.join(os.path.dirname(image_path), "combined_visualization.png")
        combined.save(output_path)
        print(f"   ✅ Saved: {output_path}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create visualization: {e}")


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("🎨 Gemini Image + Mesh Generation Example")
    print("=" * 70)
    
    # Example 1: Generate with GLB mesh
    print("\n📝 Example 1: Text-to-3D with GLB + Image")
    print("-" * 70)
    mesh_path, image_path = generate_with_image_and_mesh(
        "A futuristic robot holding a glowing orb",
        output_dir="output_glb",
        return_pickle=False
    )
    
    if image_path:
        process_image(image_path)
    
    if mesh_path:
        process_mesh(mesh_path, return_pickle=False)
    
    if image_path and mesh_path:
        create_combined_visualization(image_path, mesh_path)
    
    # Example 2: Generate with pickled mesh
    print("\n" + "=" * 70)
    print("\n📝 Example 2: Text-to-3D with Pickled Mesh + Image")
    print("-" * 70)
    mesh_path2, image_path2 = generate_with_image_and_mesh(
        "A medieval castle on a hilltop",
        output_dir="output_pickle",
        return_pickle=True
    )
    
    if image_path2:
        process_image(image_path2)
    
    if mesh_path2:
        mesh = process_mesh(mesh_path2, return_pickle=True)
        
        # Additional processing for pickled mesh
        if mesh:
            print(f"\n🔧 Performing additional transformations on pickled mesh...")
            import trimesh
            
            # Scale the mesh
            scaled_mesh = mesh.copy()
            scaled_mesh.apply_scale(1.5)
            scaled_path = mesh_path2.replace(".pkl", "_scaled.glb")
            scaled_mesh.export(scaled_path)
            print(f"   ✅ Scaled mesh saved: {scaled_path}")
            
            # Rotate the mesh
            import numpy as np
            rotated_mesh = mesh.copy()
            rotation = trimesh.transformations.rotation_matrix(
                np.radians(45), [0, 1, 0]
            )
            rotated_mesh.apply_transform(rotation)
            rotated_path = mesh_path2.replace(".pkl", "_rotated.glb")
            rotated_mesh.export(rotated_path)
            print(f"   ✅ Rotated mesh saved: {rotated_path}")
    
    if image_path2 and mesh_path2:
        create_combined_visualization(image_path2, mesh_path2)
    
    print("\n" + "=" * 70)
    print("🎉 Examples completed!")
    print("=" * 70)
    print("\n💡 What you got:")
    print("   - Gemini-generated images from your text prompts")
    print("   - 3D meshes (both GLB and pickled formats)")
    print("   - Mesh exported to multiple formats (OBJ, STL, PLY)")
    print("   - Combined visualization images")
    print("   - Transformed mesh variations (scaled, rotated)")
    print("\n📁 Check the 'output_glb' and 'output_pickle' directories!")
