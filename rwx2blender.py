import sys
import os
import re
import fileinput
import numpy as np
from copy import copy, deepcopy
from math import radians
import mathutils
import bpy
from bpy.props import *
import bmesh

bl_info = {"name": "rwx2blender",
           "author": "Julien Bardagi (Blaxar Waldarax)",
           "description": "Module to import Active Worlds RenderWare files (.rwx)",
           "version": (0, 1, 0),
           "blender": (2, 78, 0),
           "location": "File > Import...",
           "category": "Import-Export"}

class RwxState:

    def __init__(self):

        self.color = None
        self.surface = None
        self.opacity = None
        self.lightsampling = None
        self.geometrysampling = None
        self.texturemodes = None
        self.materialmodes = None
        self.texture = None
        self.transform = np.identity(4)


class RwxVertex:

    def __init__(self, x, y ,z, u = None, v = None):
        
        self.x = x
        self.y = y
        self.z = z
        self.u = u
        self.v = v

    def __str__(self):
        return "x:%s y:%s z:%s (u:%s, v:%s)" % (self.x, self.y, self.z, self.u, self.v)

    def __call__(self):
        return (self.x, self.y, self.z)

    @property
    def uv(self):
        return (self.u, self.v)
        

class RwxScope:
    
    def __init__(self, state = RwxState()):

        self.state = copy(state)
        self.vertices = []
        self.shapes = []

    def __str__(self):
        
        return "vertices:%s" % os.linesep +\
               os.linesep.join([ "-- %s" % (str(v),) for v in self.vertices ]) +\
               "{0}shapes:{0}".format(os.linesep) +\
               os.linesep.join([ "-- %s" % (str(s),) for s in self.shapes ])

    @property
    def faces(self):

        faces = []
        for shape in self.shapes: faces.extend(shape())

        return faces

    @property
    def verts(self):
        return [vert() for vert in self.vertices]

    @property
    def verts_uv(self):
        return [[vert.uv[0], 1.0-vert.uv[1] if vert.uv[1] is not None else None] for vert in self.vertices]
    
    @property
    def faces_uv(self):

        faces = self.faces
        verts = self.verts
        verts_uv = self.verts_uv
        return [ [verts_uv[face[0]], verts_uv[face[1]], verts_uv[face[2]]] for face in faces ]

    @property
    def faces_state(self):
        
        states = []
        
        for shape in self.shapes:
            for face in shape():
                states.append(shape.state)

        return states
    
class RwxClump(RwxScope):

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        self.clumps = []

    def __str__(self):

        cl = [ str(c).split(os.linesep) for c in self.clumps ]
        clumps = []
        for clump in cl: clumps.extend(clump)
        clumps = ["----%s" % str(c) for c in clumps]
        
        return "clump:%s" % os.linesep +\
               super().__str__()+os.linesep+\
               "--clumps:%s" % os.linesep +\
               os.linesep.join(clumps)

    def apply_proto(self, proto):
        
        offset = len(self.vertices)

        shapes = copy(proto.shapes)
        for shape in shapes:
            for i, vid in enumerate(shape.vertices_id):
                shape.vertices_id[i] += offset
        
        self.shapes.extend(shapes)

        for i, vert in enumerate(proto.vertices):
            mat = proto.state.transform * np.matrix([vert.x, vert.y, vert.z, 1]).reshape((4,1))
            self.vertices.append(RwxVertex(mat[0], mat[1], mat[2], u=vert.u, v=vert.v))

        
class RwxProto(RwxScope):
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        
class RwxShape:

    def __init__(self, state = RwxState()):
        
        self.state = copy(state)
        self.vertices_id = None

    def __call__(self):
        
        return []

        
class RwxTriangle(RwxShape):

    def __init__(self, v1, v2, v3, **kwargs):
        
        super().__init__(**kwargs)
        self.vertices_id = [v1, v2, v3]

    def __call__(self):
        
        return [(self.vertices_id[0]-1, self.vertices_id[1]-1, self.vertices_id[2]-1)]

    
class RwxQuad(RwxShape):

    def __init__(self, v1, v2, v3, v4, **kwargs):
        
        super().__init__(**kwargs)
        self.vertices_id = [v1, v2, v3, v4]

    def __call__(self):
        
        return [(self.vertices_id[0]-1, self.vertices_id[1]-1, self.vertices_id[2]-1),\
                (self.vertices_id[0]-1, self.vertices_id[2]-1, self.vertices_id[3]-1)]


class RwxPolygon(RwxShape):

    def __init__(self, v_id = [], **kwargs):
        
        super().__init__(**kwargs)
        self.vertices_id = v_id

    def __call__(self):

        vertices = []

        for id in range(0, len(self.vertices_id)-2):
            vertices.append((self.vertices_id[0]-1, self.vertices_id[id+1]-1, self.vertices_id[id+2]-1))
            
        return vertices


class RwxObject:

    state = None
    
    def __init__(self):
        self.protos = []
        self.clumps = []
        self.state = RwxState()

    def __str__(self):

        cl = [ str(c).split(os.linesep) for c in self.clumps ]
        clumps = []
        for clump in cl: clumps.extend(clump)
        clumps = ["----%s" % str(c) for c in clumps]
        
        return "object:%s" % os.linesep +\
               "--clumps:%s" % os.linesep +\
               os.linesep.join(clumps)


class RwxParser:

    # Begin regex list

    _integer_regex = re.compile("([0-9]+)")
    _float_regex = re.compile("([-+]?[0-9]*\\.?[0-9]+)")
    _non_comment_regex = re.compile("^(.*)#")
    _modelbegin_regex = re.compile("^ *(modelbegin).*$", re.IGNORECASE)
    _modelend_regex = re.compile("^ *(modelend).*$", re.IGNORECASE)
    _clumpbegin_regex = re.compile("^ *(clumpbegin).*$", re.IGNORECASE)
    _clumpend_regex = re.compile("^ *(clumpend).*$", re.IGNORECASE)
    _protobegin_regex = re.compile("^ *(protobegin) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _protoinstance_regex = re.compile("^ *(protoinstance) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _protoend_regex = re.compile("^ *(protoend).*$", re.IGNORECASE)
    _vertex_regex = re.compile("^ *(vertex|vertexext)(( +[-+]?[0-9]*\\.?[0-9]+){3}) *(uv(( +[-+]?[0-9]*\\.?[0-9]+){2}))?.*$", re.IGNORECASE)
    _polygon_regex = re.compile("^ *(polygon|polygonext)( +[0-9]+)(( +[0-9]+)+) ?.*$", re.IGNORECASE)
    _quad_regex = re.compile("^ *(quad|quadext)(( +([0-9]+)){4}).*$", re.IGNORECASE)
    _triangle_regex = re.compile("^ *(triangle|triangleext)(( +([0-9]+)){3}).*$", re.IGNORECASE)
    _texture_regex = re.compile("^ *(texture) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _color_regex = re.compile("^ *(color)( +[-+]?[0-9]*\\.?[0-9]+){3}.*$", re.IGNORECASE)
    _transform_regex = re.compile("^ *(transform)(( +[-+]?[0-9]*\\.?[0-9]+){16}).*$", re.IGNORECASE)
    _scale_regex = re.compile("^ *(scale)( +([0-9]+)){3}.*$", re.IGNORECASE)
    _rotate_regex = re.compile("^ *(rotate)( +[-+]?[0-9]*\\.?[0-9]+){4}.*$", re.IGNORECASE)
        
    
    def __init__(self, uri):
        
        self._rwx_clump_stack = []
        self._rwx_proto_dict = {}
        self._current_scope = None

        with fileinput.input(files=(uri,)) as f:
            res = None
            for line in f:
 
                #strip comment away
                res = self._non_comment_regex.match(line)
                if res:
                    line = res.group(1)
                
                res = self._modelbegin_regex.match(line)
                if res:
                    self._rwx_clump_stack.append(RwxObject())
                    self._current_scope = self._rwx_clump_stack[-1]
                    continue
                
                res = self._clumpbegin_regex.match(line)
                if res:
                    rwx_clump = RwxClump(state = self._current_scope.state)
                    self._rwx_clump_stack[-1].clumps.append(rwx_clump)
                    self._rwx_clump_stack.append(rwx_clump)
                    self._current_scope = rwx_clump
                    continue

                res = self._clumpend_regex.match(line)
                if res:
                    self._rwx_clump_stack.pop()
                    self._current_scope = self._rwx_clump_stack[-1]
                    continue

                res = self._protobegin_regex.match(line)
                if res:
                    name = res.group(2)
                    self._rwx_proto_dict[name] = RwxScope(state = self._current_scope.state)
                    self._current_scope = self._rwx_proto_dict[name]
                    continue

                res = self._protoend_regex.match(line)
                if res:
                    self._current_scope = self._rwx_clump_stack[0]
                    continue

                res = self._protoinstance_regex.match(line)
                if res:
                    name = res.group(2)
                    self._current_scope.apply_proto(self._rwx_proto_dict[name])
                    continue

                res = self._texture_regex.match(line)
                if res:
                    self._current_scope.state.texture = None if res.group(2).lower() == "null" else res.group(2)
                    continue
                    
                res = self._triangle_regex.match(line)
                if res:
                    v_id = [ int(x) for x in self._integer_regex.findall(res.group(2)) ]
                    self._current_scope.shapes.append(RwxTriangle(v_id[0], v_id[1], v_id[2],\
                                                                  state=self._current_scope.state))
                    continue

                res = self._quad_regex.match(line)
                if res:
                    v_id = [ int(x) for x in self._integer_regex.findall(res.group(2)) ]
                    self._current_scope.shapes.append(RwxQuad(v_id[0], v_id[1], v_id[2], v_id[3],\
                                                              state=self._current_scope.state))
                    continue
                
                res = self._polygon_regex.match(line)
                if res:
                    v_len = int(self._integer_regex.findall(res.group(2))[0])
                    v_id = [ int(x) for x in self._integer_regex.findall(res.group(3)) ]
                    self._current_scope.shapes.append(RwxPolygon(v_id[0:v_len],\
                                                                 state=self._current_scope.state))
                    continue
                
                res = self._vertex_regex.match(line)
                if res:
                    vprops = [ float(x) for x in self._float_regex.findall(res.group(2)) ]
                    if res.group(5):
                        vprops.extend([ float(x) for x in self._float_regex.findall(res.group(5)) ])
                        self._current_scope.vertices.append(RwxVertex(vprops[0], vprops[1], vprops[2], u = vprops[3], v = vprops[4]))
                    else: self._current_scope.vertices.append(RwxVertex(vprops[0], vprops[1], vprops[2]))
                    continue

                res = self._transform_regex.match(line)
                if res:
                    tprops = [ float(x) for x in self._float_regex.findall(res.group(2)) ]
                    if len(tprops) == 16: self._current_scope.state.transform = np.matrix(tprops).reshape((4,4)).T

                res = self._scale_regex.match(line)
                if res:
                    sprops = [ float(x) for x in self._float_regex.findall(res.group(2)) ]
                    if len(sprops) == 16: self._current_scope.state.transform *= np.matrix(tprops).reshape((4,4)).T

    def __call__(self):
        return self._rwx_clump_stack[0]


def add_attr_recursive(clump, name):

    attr = []
    attr.extend(getattr(clump, name))
    
    for c in clump.clumps:
        attr.extend(add_attr_recursive(c, name))

    return attr


def add_vertices_recursive(clump, transform = np.identity(4)):

    vertices = []
    transform = transform * clump.state.transform
    for v in clump.verts:
        vert = transform * np.matrix([v[0], v[1], v[2], 1]).reshape((4,1))
        vertices.append((vert[0], vert[1], vert[2]))

    for c in clump.clumps:
        vertices.extend(add_vertices_recursive(c))

    return vertices


def add_faces_recursive(clump, offset=0):

    faces = []
    tmp_faces = clump.faces
    for tmp_face in tmp_faces:
        faces.append((tmp_face[0]+offset, tmp_face[1]+offset, tmp_face[2]+offset))
    
    offset += len(clump.verts)
    
    for c in clump.clumps:
        tmp_faces, offset = add_faces_recursive(c, offset)
        faces.extend(tmp_faces)

    return faces, offset


def make_materials_recursive(ob, clump, folder, extension = "jpg"):

    for shape in clump.shapes:
        # Get material

        if shape.state.texture:
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
        
class Rwx2BlenderOperator(bpy.types.Operator):

    bl_idname = "object.rwx2blender"
    bl_label = "Import"
    bl_options = {'REGISTER'}

    filepath= StringProperty(
        name="File Path",
        description="Filepath used for importing RenderWare .rwx file",
        maxlen=1024,
        default="",
        subtype='FILE_PATH')
    texturepath= StringProperty(
        name="Texture Path",
        description="Path to the texture directory",
        maxlen=1024,
        default="",
        subtype='DIR_PATH')

    def invoke(self, context, event):
        wm = bpy.context.window_manager
        wm.fileselect_add(self)

        return {'RUNNING_MODAL'}

    def execute(self, context):
        
        filepath = self.filepath
        texturepath = self.texturepath

        if texturepath is None or texturepath == "":
            texturepath = os.path.join(os.path.split(os.path.dirname(filepath))[0], "textures")
            if os.path.isdir(texturepath):
                self.report({'WARNING'}, "No texture directory specified, assuming %s" % texturepath)
            else:
                self.report({'ERROR'}, "No texture directory specified and could not guess any.")
                return {'CANCELLED'}

        try:
            parser = RwxParser(filepath)
            rwx_object = parser()
        except Exception as exc:
            self.report({'ERROR'}, "Could not parse input file (likely not a proper .rwx).")
            return {'CANCELLED'}

        if len(rwx_object.clumps) == 0:
            self.report({'ERROR'}, "No clump registered after parsing the input file (likely not a proper .rwx).")
            return {'CANCELLED'}

        verts = add_vertices_recursive(rwx_object.clumps[0])
        edges = []
        faces, offset = add_faces_recursive(rwx_object.clumps[0])
        faces_state = add_attr_recursive(rwx_object.clumps[0], "faces_state")
        faces_uv = add_attr_recursive(rwx_object.clumps[0], "faces_uv")

        mesh = bpy.data.meshes.new('Mesh')
        ob = bpy.data.objects.new('Object', mesh)
        
        make_materials_recursive(ob, rwx_object.clumps[0], texturepath)

        # Create mesh from given verts, edges, faces. Either edges or
        # faces should be [], or you ask for problems
        mesh.from_pydata(verts, edges, faces)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        uv_layer = bm.loops.layers.uv.verify()
        bm.faces.layers.tex.verify()  # currently blender needs both layers.

        # adjust UVs
        for i, f in enumerate(bm.faces):
            if faces_state[i].texture:
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
        
        return {'FINISHED'}

def import_menu_func_rwx(self, context):
    op = self.layout.operator(Rwx2BlenderOperator.bl_idname, text="Active Worlds RenderWare file (.rwx)")

def register():
    bpy.utils.register_class(Rwx2BlenderOperator)
    bpy.types.INFO_MT_file_import.append(import_menu_func_rwx)

def unregister():
    bpy.utils.unregister_class(Rwx2BlenderOperator)
    bpy.types.INFO_MT_file_import.remove(import_menu_func_rwx)

if __name__ == "__main__":
    register()

