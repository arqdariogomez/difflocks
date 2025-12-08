import bpy
import sys
import numpy as np
import os

# --- ARGS PARSING ---
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    sys.exit(1)

input_npz = argv[0]
output_base = argv[1]

print(f"🔄 DiffLocks Converter Processing: {input_npz}")

# Limpiar escena por defecto
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- CONFIGURACIÓN (Igual que tu Addon) ---
SCALE_FACTOR = 1.0
ROTATE_X_90 = True
CONVERT_TO_HAIR = True
USE_VERTEX_COLORS = True

def create_safe_material(name, has_colors=False):
    """Lógica idéntica a tu addon para crear el material"""
    mat = bpy.data.materials.new(name=f"{name}_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    shader = nodes.new(type='ShaderNodeBsdfHairPrincipled')
    shader.location = (300, 300)
    
    # Preset 'BLONDE' por defecto como en tu script
    # presets = {'BLACK': (0.95, 0.1), 'BROWN': (0.65, 0.5), 'BLONDE': (0.25, 0.3), 'RED': (0.60, 0.95)}
    m, r = (0.25, 0.3) # Blonde
    
    if has_colors and USE_VERTEX_COLORS:
        try:
            attr_node = nodes.new(type='ShaderNodeAttribute')
            attr_node.attribute_name = "DiffLocks_Color"
            attr_node.location = (0, 400)
            # Conectar color
            links.new(attr_node.outputs["Color"], shader.inputs[0]) # Color input
            m = 0.0 # Anular melanina si hay color
        except: pass

    # Configurar inputs básicos
    shader.inputs["Melanin"].default_value = m
    shader.inputs["Redness"].default_value = r
    shader.inputs["Roughness"].default_value = 0.5
    shader.inputs["Coat"].default_value = 0.1

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (600, 300)
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    # 1. CARGAR DATOS
    data = np.load(input_npz)
    positions = data['positions']
    colors = data.get('colors', None)
    radii = data.get('radii', None)
    
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    print(f"🚀 Processing {num_strands} strands...")

    # 2. PREPARAR GEOMETRÍA (Tu lógica exacta)
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        # Rotación manual de arrays para evitar operadores de objeto
        # x, y, z -> x, z, -y (Blender Z-up standard conversion)
        # Tu script hacía: flat_pos = flat_pos[:, [0, 2, 1]]; flat_pos[:, 1] *= -1
        flat_pos_rot = flat_pos[:, [0, 2, 1]]
        flat_pos_rot[:, 1] *= -1
        flat_pos = flat_pos_rot
        
    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos
    points_4d[:, 3] = 1.0

    # 3. CREAR CURVA LEGACY
    curve_data = bpy.data.curves.new(name="DiffLocks_Temp", type='CURVE')
    curve_data.dimensions = '3D'
    
    for i in range(num_strands):
        s = curve_data.splines.new('POLY')
        s.points.add(pts_per_strand - 1)
        start = i * pts_per_strand
        end = start + pts_per_strand
        s.points.foreach_set('co', points_4d[start:end].ravel())
        
    temp_obj = bpy.data.objects.new("DiffLocks_Temp", curve_data)
    bpy.context.collection.objects.link(temp_obj)
    
    # Seleccionar para operaciones
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = temp_obj
    temp_obj.select_set(True)
    
    final_obj = temp_obj

    # 4. CONVERTIR A HAIR CURVES (Moderno)
    if CONVERT_TO_HAIR:
        print("✨ Converting to Modern Hair Curves...")
        bpy.ops.object.convert(target='CURVES', keep_original=False)
        final_obj = bpy.context.active_object
        final_obj.name = "DiffLocks_Hair"
        
        # Atributos: Radios
        if radii is not None:
            r_flat = radii.reshape(-1) * SCALE_FACTOR
            if len(final_obj.data.attributes['radius'].data) == len(r_flat):
                final_obj.data.attributes['radius'].data.foreach_set('value', r_flat.astype(np.float32))
        else:
            # Default radius
            total_pts = len(final_obj.data.points)
            defaults = np.full(total_pts, 0.003 * SCALE_FACTOR, dtype=np.float32)
            final_obj.data.attributes['radius'].data.foreach_set('value', defaults)

        # Atributos: Colores
        if colors is not None and USE_VERTEX_COLORS:
            if "DiffLocks_Color" not in final_obj.data.attributes:
                attr = final_obj.data.attributes.new(name="DiffLocks_Color", type='FLOAT_COLOR', domain='POINT')
            else:
                attr = final_obj.data.attributes["DiffLocks_Color"]
            
            c_flat = colors.reshape(-1, 3)
            rgba = np.ones((len(c_flat), 4), dtype=np.float32)
            rgba[:, :3] = c_flat
            attr.data.foreach_set('color', rgba.ravel())

    # 5. MATERIAL
    mat = create_safe_material("DiffLocks", has_colors=(colors is not None))
    if final_obj.data.materials: final_obj.data.materials[0] = mat
    else: final_obj.data.materials.append(mat)

    # ========================================================
    # EXPORTACIÓN
    # ========================================================
    
    # A) GUARDAR .BLEND (Editable, idéntico a tu importador)
    blend_path = f"{output_base}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"✅ Saved: {blend_path}")
    
    # B) EXPORTAR ALEMBIC (Soporta curvas nativas)
    abc_path = f"{output_base}.abc"
    bpy.ops.wm.alembic_export(filepath=abc_path, selected=True)
    print(f"✅ Exported: {abc_path}")
    
    # C) EXPORTAR GLB (Para la Web)
    # Las curvas de pelo modernas NO se ven en visores GLB estándar.
    # Necesitamos convertirlas a Malla con grosor solo para este archivo.
    
    print("🔨 Converting to Mesh for GLB preview...")
    
    # Agregar modificador Geometry Nodes simple para darle grosor
    mod = final_obj.modifiers.new("GeoNodes", "NODES")
    node_group = bpy.data.node_groups.new("HairToMesh", "GeometryNodeTree")
    mod.node_group = node_group
    
    nodes = node_group.nodes
    links = node_group.links
    
    # Nodos: Input -> Curve to Mesh (con Circle Profile) -> Output
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    
    curve_to_mesh = nodes.new("GeometryNodeCurveToMesh")
    circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.inputs["Radius"].default_value = 0.002 # Grosor visual
    circle.inputs["Resolution"].default_value = 3 # Bajo poli
    
    links.new(input_node.outputs[0], curve_to_mesh.inputs["Curve"])
    links.new(circle.outputs["Curve"], curve_to_mesh.inputs["Profile Curve"])
    links.new(curve_to_mesh.outputs["Mesh"], output_node.inputs[0])
    
    # Aplicar y convertir a malla
    bpy.ops.object.convert(target='MESH')
    
    glb_path = f"{output_base}.glb"
    bpy.ops.export_scene.gltf(filepath=glb_path, export_format='GLB', use_selection=True)
    print(f"✅ Exported: {glb_path}")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)