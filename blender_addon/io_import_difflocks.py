import bpy
import os
import time
import traceback
import numpy as np
import sys
from bpy_extras.io_utils import ImportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty, EnumProperty, IntProperty
from bpy.types import Operator

bl_info = {
    "name": "Import DiffLocks",
    "author": "Dario Gomez",
    "version": (1, 0),
    "blender": (4, 1, 0),
    "location": "File > Import",
    "description": "Import with controls for both Strand Amount and Point Resolution",
    "category": "Import-Export",
}

class ImportDiffLocksDual(Operator, ImportHelper):
    bl_idname = "import_curve.difflocks_v37"
    bl_label = "Import DiffLocks (Dual Control)"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.npz", options={'HIDDEN'})
    
    scale_factor: FloatProperty(
        name="Global Scale", 
        description="Scale multiplier (1.0 = Original size)",
        default=1.0, min=0.01, max=100.0
    )
    
    rotate_x_90: BoolProperty(
        name="Rotate 90° (Z-Up)", 
        default=True
    )
    
    convert_to_hair: BoolProperty(
        name="Convert to Hair Curves", 
        default=True
    )
    
    # --- CONTROL 1: QUANTITY ---
    strand_count_pct: IntProperty(
        name="Strand Amount %",
        description="How many hairs to import. 10% is great for quick testing/guides.",
        default=100, min=1, max=100
    )

    # --- CONTROL 2: QUALITY ---
    point_resolution_pct: IntProperty(
        name="Point Resolution %",
        description="Smoothness per hair. 20% keeps the shape but removes extra vertices.",
        default=100, min=5, max=100
    )
    
    hair_preset: EnumProperty(
        name="Base Color",
        items=[
            ('BLONDE', "Blonde", "Light/Nordic"), 
            ('BROWN', "Brown", "Standard Brunette"), 
            ('BLACK', "Black", "Dark/Asian/African"), 
            ('RED', "Red", "Ginger/Celtic")
        ],
        default='BLONDE', 
    )
    
    use_vertex_colors: BoolProperty(
        name="Use Color Data", 
        default=True
    )

    def execute(self, context):
        if not os.path.exists(self.filepath): return {'CANCELLED'}
        context.window.cursor_modal_set('WAIT')
        try:
            return self.read_npz_file(context)
        except Exception as e:
            self.report({'ERROR'}, f"Error: {e}")
            traceback.print_exc()
            return {'CANCELLED'}
        finally:
            context.window.cursor_modal_set('DEFAULT')

    def create_safe_material(self, name, hair_data=None):
        mat_name = f"{name}_Mat"
        if mat_name in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials[mat_name])
            
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()
        
        shader = nodes.new(type='ShaderNodeBsdfHairPrincipled')
        shader.location = (300, 300)
        
        presets = {'BLACK': (0.95, 0.1), 'BROWN': (0.65, 0.5), 'BLONDE': (0.25, 0.3), 'RED': (0.60, 0.95)}
        m, r = presets.get(self.hair_preset, (0.65, 0.5))
        
        used_attributes = False
        if hair_data and 'colors' in hair_data and self.use_vertex_colors:
            try:
                attr_node = nodes.new(type='ShaderNodeAttribute')
                attr_node.attribute_name = "DiffLocks_Color"
                attr_node.location = (0, 400)
                for inp in shader.inputs:
                    if "color" in inp.name.lower() and "random" not in inp.name.lower():
                        links.new(attr_node.outputs["Color"], inp)
                        m = 0.0; used_attributes = True; break
            except: pass

        for inp in shader.inputs:
            n = inp.name.lower()
            if not used_attributes: 
                if "melanin" in n and "redness" not in n: inp.default_value = m
                if "redness" in n: inp.default_value = r
            if "roughness" in n and "radial" not in n: inp.default_value = 0.5
            if "coat" in n: inp.default_value = 0.1

        out = nodes.new(type='ShaderNodeOutputMaterial')
        out.location = (600, 300)
        links.new(shader.outputs[0], out.inputs[0])
        return mat

    def read_npz_file(self, context, filepath=None):
        t_total_start = time.time()
        filepath = filepath or self.filepath
        
        try: import numpy as np
        except: raise ImportError("Numpy not installed.")

        print(f"\n{'='*40}")
        print(f"📂 Reading: {filepath}")
        data = np.load(filepath)
        positions = data['positions']
        colors = data.get('colors', None)
        radii = data.get('radii', None)
        
        total_available = positions.shape[0]
        
        # --- OPTIMIZATION 1: STRAND COUNT (Quantity) ---
        if self.strand_count_pct < 100:
            limit = int(total_available * (self.strand_count_pct / 100.0))
            print(f"✂️  Quantity Limit: {self.strand_count_pct}% ({limit} / {total_available} strands)")
            
            # Slice Dimension 0 (Strands)
            positions = positions[:limit]
            if radii is not None:
                if radii.ndim >= 1: radii = radii[:limit]
            if colors is not None:
                if colors.ndim >= 1: colors = colors[:limit]
        
        # --- OPTIMIZATION 2: POINT RESOLUTION (Quality) ---
        if self.point_resolution_pct < 100:
            factor = self.point_resolution_pct / 100.0
            step = int(1.0 / factor)
            if step < 1: step = 1
            print(f"✂️  Quality Reduction: {self.point_resolution_pct}% (1 out of {step} points)")
            
            # Slice Dimension 1 (Points per strand)
            positions = positions[:, ::step, :]
            if radii is not None and radii.ndim == 2: radii = radii[:, ::step]
            if colors is not None and colors.ndim == 3: colors = colors[:, ::step, :]
        
        hair_data_dict = {'colors': colors} if colors is not None else None
        num_strands = int(positions.shape[0])
        pts_per_strand = int(positions.shape[1])
        
        print(f"🚀 Processing {num_strands:,} strands ({pts_per_strand} pts each)...")

        # 1. GEOMETRY PREP
        flat_pos = positions.reshape(-1, 3) * self.scale_factor
        if self.rotate_x_90:
            flat_pos = flat_pos[:, [0, 2, 1]]
            flat_pos[:, 1] *= -1
            
        points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
        points_4d[:, :3] = flat_pos
        points_4d[:, 3] = 1.0

        # 2. LEGACY CURVE BUILD (With Feedback)
        print("🔨 Building Geometry: ", end="", flush=True)
        curve_data = bpy.data.curves.new(name="DiffLocks_Temp", type='CURVE')
        curve_data.dimensions = '3D'
        
        report_interval = max(1, num_strands // 10)
        
        for i in range(num_strands):
            s = curve_data.splines.new('POLY')
            s.points.add(pts_per_strand - 1)
            start = i * pts_per_strand
            end = start + pts_per_strand
            s.points.foreach_set('co', points_4d[start:end].ravel())
            
            if i > 0 and i % report_interval == 0:
                percent = int((i / num_strands) * 100)
                print(f"{percent}%...", end=" ", flush=True)
        
        print("100% Done.")

        # 3. OBJECT CREATION
        temp_obj = bpy.data.objects.new("DiffLocks_Temp", curve_data)
        context.collection.objects.link(temp_obj)
        bpy.ops.object.select_all(action='DESELECT')
        temp_obj.select_set(True)
        context.view_layer.objects.active = temp_obj
        
        # 4. CONVERSION
        final_obj = temp_obj
        
        if self.convert_to_hair:
            print("✨ Converting to Modern Hair Curves...", end=" ", flush=True)
            t_conv = time.time()
            try:
                bpy.ops.object.convert(target='CURVES', keep_original=False)
                final_obj = context.active_object
                final_obj.name = "DiffLocks_Hair"
                print(f"Done ({time.time() - t_conv:.2f}s)")
                
                # Attributes
                if radii is not None:
                    r_flat = radii.reshape(-1) * self.scale_factor
                    if len(final_obj.data.attributes['radius'].data) == len(r_flat):
                        final_obj.data.attributes['radius'].data.foreach_set('value', r_flat.astype(np.float32))
                else:
                    total_pts = len(final_obj.data.points)
                    defaults = np.full(total_pts, 0.003 * self.scale_factor, dtype=np.float32)
                    final_obj.data.attributes['radius'].data.foreach_set('value', defaults)

                if colors is not None and self.use_vertex_colors:
                    try:
                        if "DiffLocks_Color" not in final_obj.data.attributes:
                            attr = final_obj.data.attributes.new(name="DiffLocks_Color", type='FLOAT_COLOR', domain='POINT')
                        else: attr = final_obj.data.attributes["DiffLocks_Color"]
                        c_flat = colors.reshape(-1, 3)
                        rgba = np.ones((len(c_flat), 4), dtype=np.float32)
                        rgba[:, :3] = c_flat
                        attr.data.foreach_set('color', rgba.ravel())
                    except: pass

            except Exception as e:
                print(f"\n⚠️ Conversion failed: {e}")

        # 5. FINISH
        print("🎨 Assigning Material...")
        mat = self.create_safe_material("DiffLocks", hair_data_dict)
        if final_obj.data.materials: final_obj.data.materials[0] = mat
        else: final_obj.data.materials.append(mat)
        
        try: final_obj.data.surface_uv_map = "UVMap" 
        except: pass

        try:
            for area in context.screen.areas:
                if area.type == 'VIEW_3D':
                    area.spaces.active.shading.type = 'MATERIAL'
                    ctx = context.copy(); ctx['area'] = area
                    with context.temp_override(**ctx): bpy.ops.view3d.view_selected()
                    break
        except: pass

        print(f"✅ TOTAL TIME: {time.time() - t_total_start:.2f}s")
        print("="*40)
        return {'FINISHED'}

def register():
    bpy.utils.register_class(ImportDiffLocksDual)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)

def unregister():
    bpy.utils.unregister_class(ImportDiffLocksDual)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func)

def menu_func(self, context):
    self.layout.operator(ImportDiffLocksDual.bl_idname, text="DiffLocks Hair (.npz)")

if __name__ == "__main__":
    try: unregister()
    except: pass
    register()
    bpy.ops.import_curve.difflocks_v37('INVOKE_DEFAULT')