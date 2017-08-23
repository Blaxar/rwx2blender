import sys
import os
from math import radians
sys.path.append("/home/blax/Projects/rwx2blender")

from rwx import RwxParser
import bpy
import bmesh


def add_attr_recursive(clump, name):

    attr = []
    attr.extend(getattr(clump, name))
    
    for c in clump.clumps:
        attr.extend(add_attr_recursive(c, name))

    return attr


def make_materials_recursive(ob, clump, folder, extension = "jpg"):

    for shape in clump.shapes:
        # Get material
        
        mat = bpy.data.materials.get(shape.state.texture)
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name=shape.state.texture)
            tex = bpy.data.textures.new(shape.state.texture, type = 'IMAGE')
            tex.image = bpy.data.images.load(os.path.join(folder, "%s.%s" % (shape.state.texture, extension)))
            mtex = mat.texture_slots.add()
            mtex.texture = tex
            mtex.texture_coords = 'UV'
            mtex.use_map_color_diffuse = True 
            mtex.use_map_color_emission = True 
            mtex.emission_color_factor = 1.0
            mtex.use_map_density = True 
            mtex.mapping = 'FLAT'

        unassigned = True
        for material in ob.data.materials:
            if material.name == shape.state.texture:
                unassigned = False
        
        if unassigned: ob.data.materials.append(mat)
        
    for sub_clump in clump.clumps:
        make_materials_recursive(ob, sub_clump, folder, extension)
    

parser = RwxParser("/home/blax/Projects/rwx2blender/tracteur1.rwx")
rwx_object = parser()

verts = add_attr_recursive(rwx_object.clumps[0], "verts")
edges = []
faces = add_attr_recursive(rwx_object.clumps[0], "faces")
faces_state = add_attr_recursive(rwx_object.clumps[0], "faces_state")
faces_uv = add_attr_recursive(rwx_object.clumps[0], "faces_uv")

mesh = bpy.data.meshes.new('Mesh')
ob = bpy.data.objects.new('Object', mesh)
make_materials_recursive(ob, rwx_object.clumps[0], "/home/blax/Stockage/village2/textures")

# Create mesh from given verts, edges, faces. Either edges or
# faces should be [], or you ask for problems
mesh.from_pydata(verts, edges, faces)
bm = bmesh.new()
bm.from_mesh(mesh)
uv_layer = bm.loops.layers.uv.verify()
bm.faces.layers.tex.verify()  # currently blender needs both layers.

# adjust UVs
for i, f in enumerate(bm.faces):
    f.material_index = ob.data.materials.keys().index(faces_state[i].texture)
    for j, l in enumerate(f.loops):
        l[uv_layer].uv = faces_uv[i][j]


bm.to_mesh(mesh)
bm.free()

mesh.use_auto_smooth = True
mesh.auto_smooth_angle = 3.14/3.0 
mesh.calc_normals()

ob.location = (0,0,0)
ob.scale = (10,10,10)
ob.rotation_euler = (radians(90), 0, 0)
ob.show_name = True 
# Link object to scene
bpy.context.scene.objects.link(ob)

# Update mesh with new data
mesh.update(calc_edges=True)
