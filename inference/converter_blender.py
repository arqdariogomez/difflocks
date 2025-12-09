
import bpy
import sys
import numpy as np
import os

# --- ARGS PARSING ---
# Usage: blender ... -- input.npz output_base format1 format2 ...
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
    input_npz = args[0]
    output_base = args[1]
    requested_formats = args[2:] 
else:
    print("❌ Args missing")
    sys.exit(1)

print(f"🔄 Blender Processing: {input_npz}")
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- CONFIG ---
SCALE_FACTOR = 1.0
ROTATE_X_90 = True # Fix Y-Up to Z-Up

def create_material():
    mat = bpy.data.materials.new(name="DiffLocks_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    shader = nodes.new(type='ShaderNodeBsdfPrincipled') 
    shader.inputs["Base Color"].default_value = (0.05, 0.03, 0.01, 1.0) # Dark Brown
    shader.inputs["Roughness"].default_value = 0.5
    out = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    # LOAD
    data = np.load(input_npz)
    positions = data['positions']
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    # TRANSFORM
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        flat_pos = flat_pos[:, [0, 2, 1]] # Swap Y and Z
        flat_pos[:, 1] *= -1 # Invert Y

    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos
    points_4d[:, 3] = 1.0

    # CURVE
    curve_data = bpy.data.curves.new(name="Hair", type='CURVE')
    curve_data.dimensions = '3D'
    
    for i in range(num_strands):
        s = curve_data.splines.new('POLY') 
        s.points.add(pts_per_strand - 1)
        start = i * pts_per_strand
        end = start + pts_per_strand
        s.points.foreach_set('co', points_4d[start:end].ravel())
        
    obj = bpy.data.objects.new("DiffLocks_Hair", curve_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # CONVERT TO GEOMETRY
    bpy.ops.object.convert(target='CURVES', keep_original=False)
    obj = bpy.context.active_object
    
    # MATERIAL
    mat = create_material()
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)

    # EXPORT
    if 'blend' in requested_formats:
        out = f"{output_base}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
    
    if 'abc' in requested_formats:
        out = f"{output_base}.abc"
        bpy.ops.wm.alembic_export(filepath=out, selected=True)

    if 'usd' in requested_formats:
        out = f"{output_base}.usd"
        bpy.ops.wm.usd_export(filepath=out, selected_objects_only=True)
        
    if 'obj' in requested_formats:
        out = f"{output_base}.obj"
        # Fallback for OBJ if needed via Blender
        bpy.ops.wm.obj_export(filepath=out, export_selected_objects=True)

    print("✅ SUCCESS")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
