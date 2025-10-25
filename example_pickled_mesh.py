#!/usr/bin/env python3
"""
Example: Using pickled mesh objects with the API

This script demonstrates how to:
1. Request a pickled mesh object instead of GLB
2. Load and manipulate the pickled mesh
3. Convert it to other formats if needed
"""

import pickle
import requests
import trimesh


def generate_pickled_mesh(text_prompt, output_file="mesh.pkl"):
    """
    Generate a 3D mesh and receive it as a pickled object.
    
    Args:
        text_prompt: Text description of the object
        output_file: Where to save the pickled mesh
    
    Returns:
        The loaded trimesh object if successful, None otherwise
    """
    api_url = "http://localhost:8080/generate"
    
    payload = {
        "text": text_prompt,
        "seed": 1234,
        "octree_resolution": 128,
        "num_inference_steps": 5,
        "guidance_scale": 5.0,
        "return_pickle": True,  # Request pickled mesh
        "return_gemini_image": False  # No image needed
    }
    
    print(f"Generating pickled mesh from: '{text_prompt}'")
    
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        
        if response.status_code == 200:
            # Save the pickled mesh
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✅ Pickled mesh saved to {output_file}")
            
            # Load and return the mesh
            with open(output_file, 'rb') as f:
                mesh = pickle.load(f)
            print(f"📦 Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            return mesh
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def manipulate_mesh_example(mesh):
    """
    Example of manipulating a trimesh object.
    
    Args:
        mesh: A trimesh object
    """
    print("\n🔧 Mesh manipulation examples:")
    
    # Get mesh properties
    print(f"  - Volume: {mesh.volume:.2f}")
    print(f"  - Surface area: {mesh.area:.2f}")
    print(f"  - Center of mass: {mesh.center_mass}")
    print(f"  - Bounds: {mesh.bounds}")
    
    # Apply transformations
    print("\n  Applying transformations...")
    
    # Scale the mesh
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(2.0)
    print(f"  - Scaled 2x: {len(scaled_mesh.vertices)} vertices")
    
    # Rotate the mesh
    import numpy as np
    rotated_mesh = mesh.copy()
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.radians(45),
        direction=[0, 1, 0],
        point=mesh.centroid
    )
    rotated_mesh.apply_transform(rotation_matrix)
    print(f"  - Rotated 45° around Y-axis")
    
    # Subdivide the mesh
    subdivided_mesh = mesh.copy()
    subdivided_mesh = subdivided_mesh.subdivide()
    print(f"  - Subdivided: {len(subdivided_mesh.vertices)} vertices, {len(subdivided_mesh.faces)} faces")
    
    return scaled_mesh, rotated_mesh, subdivided_mesh


def export_to_formats(mesh, base_name="output"):
    """
    Export the mesh to various formats.
    
    Args:
        mesh: A trimesh object
        base_name: Base name for output files
    """
    print(f"\n💾 Exporting to different formats...")
    
    formats = {
        'glb': 'GLB (Binary glTF)',
        'obj': 'Wavefront OBJ',
        'stl': 'STL (for 3D printing)',
        'ply': 'PLY (Polygon File Format)',
        'off': 'OFF (Object File Format)'
    }
    
    for fmt, description in formats.items():
        try:
            filename = f"{base_name}.{fmt}"
            mesh.export(filename)
            print(f"  ✅ {description}: {filename}")
        except Exception as e:
            print(f"  ❌ Failed to export {fmt}: {e}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("🎨 Pickled Mesh Example")
    print("=" * 60)
    
    # Generate a pickled mesh
    mesh = generate_pickled_mesh(
        "A cute robot toy",
        "robot_mesh.pkl"
    )
    
    if mesh:
        # Manipulate the mesh
        scaled, rotated, subdivided = manipulate_mesh_example(mesh)
        
        # Export to various formats
        export_to_formats(mesh, "robot")
        
        # You can also save modified meshes
        print(f"\n💾 Saving modified versions...")
        scaled.export("robot_scaled.glb")
        print(f"  ✅ Saved scaled version: robot_scaled.glb")
        
        rotated.export("robot_rotated.glb")
        print(f"  ✅ Saved rotated version: robot_rotated.glb")
        
        subdivided.export("robot_subdivided.glb")
        print(f"  ✅ Saved subdivided version: robot_subdivided.glb")
        
        # Save the mesh back as pickle for later use
        with open("robot_mesh_modified.pkl", 'wb') as f:
            pickle.dump(scaled, f)
        print(f"  ✅ Saved modified mesh as pickle: robot_mesh_modified.pkl")
        
        print("\n" + "=" * 60)
        print("🎉 Example completed!")
        print("=" * 60)
        print("\n💡 Benefits of using pickled meshes:")
        print("  - Direct access to trimesh object")
        print("  - No export/import overhead")
        print("  - Preserves all mesh properties")
        print("  - Easy to manipulate programmatically")
        print("  - Can be converted to any format later")
