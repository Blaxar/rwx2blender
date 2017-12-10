import sys
import os
import re
import fileinput
from copy import copy
from math import radians
import mathutils as mu
from hashlib import md5

in_blender = None

try:
    import bpy
    from bpy.props import *
    import bmesh
    from bmesh.ops import edgeloop_fill
except ModuleNotFoundError as mnf:
    in_blender = False
else:
    in_blender = True

from traceback import print_exc
from enum import Enum

bl_info = {"name": "rwx2blender",
           "author": "Julien Bardagi (Blaxar Waldarax)",
           "description": "Add-on to import Active Worlds RenderWare scripts (.rwx)",
           "version": (0, 2, 1),
           "blender": (2, 79, 0),
           "location": "File > Import...",
           "category": "Import-Export"}

class LightSampling(Enum):
    FACET = 1
    VERTEX = 2

class GeometrySampling(Enum):
    POINTCOULD = 1
    WIREFRAME = 2
    SOLID = 3

class TextureMode(Enum):
    LIT = 1
    FORESHORTEN = 2
    FILTER = 3

class MaterialMode(Enum):
    NONE = None
    NULL = 1
    DOUBLE = 2

class RwxState:

    def __init__(self):

        # Material related properties start here
        self.color = (0.0, 0.0, 0.0) # Red, Green, Blue
        self.surface = (0.0, 0.0, 0.0) # Ambience, Diffusion, Specularity
        self.opacity = 1.0
        self.lightsampling = LightSampling.FACET
        self.geometrysampling = GeometrySampling.SOLID
        self.texturemodes = [TextureMode.LIT,] # There's possibly more than one mode enabled at a time (hence why we use an array)
        self.materialmode = MaterialMode.NONE # Neither NULL nor DOUBLE: we only render one side of the polygon
        self.texture = None
        # End of material related properties
        
        self.transform = mu.Matrix.Identity(4)
        
    @property
    def mat_signature(self):

        h = md5()

        sign = [("%.3f" % x) for x in self.color]
        sign.extend([("%.3f" % x) for x in self.surface])
        sign.append("%.3f" % self.opacity)
        sign.append(self.lightsampling.name)
        sign.append(self.geometrysampling.name)
        sign.extend([x.name for x in sorted(self.texturemodes)])
        sign.append(self.materialmode.name)
        
        h.update("".join([str(x) for x in sign]).replace(".","").lower().encode("utf-8"))
        return "_".join([str(self.texture), h.hexdigest()[:10]])

    def __str__(self):
        return self.mat_signature()

    def __repr__(self):
        return "<RwxState: %s>" % self.mat_signature
        
    

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
        for shape in self.shapes:
            if not isinstance(shape, RwxPolygon): faces.extend(shape())

        return faces

    @property
    def polys(self):

        polys = []
        for shape in self.shapes:
            if isinstance(shape, RwxPolygon): polys.append(shape())

        return polys

    @property
    def verts(self):
        return [vert() for vert in self.vertices]

    @property
    def verts_uv(self):
        return [[vert.uv[0], 1.0-vert.uv[1] if vert.uv[1] is not None else None] for vert in self.vertices]
    
    @property
    def faces_uv(self):

        faces = self.faces
        verts_uv = self.verts_uv

        return [ [verts_uv[face[0]], verts_uv[face[1]], verts_uv[face[2]]] for face in faces ]

    @property
    def polys_uv(self):

        polys = self.polys
        verts_uv = self.verts_uv

        return [ [ verts_uv[edge[0]] for edge in enumerate(poly) ] for poly in polys ]

    @property
    def faces_state(self):
        
        states = []
        
        for shape in self.shapes:
            for face in shape():
                if not isinstance(shape, RwxPolygon): states.append(shape.state)

        return states

    @property
    def polys_state(self):
        
        states = []
        
        for shape in self.shapes:
                if isinstance(shape, RwxPolygon): states.append(shape.state)

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
            mat = proto.state.transform * mu.Vector([vert.x, vert.y, vert.z, 1])
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

        edges = []
        vertices_id = self.vertices_id

        for id in range(0, len(vertices_id)-1):
            if vertices_id[id] != vertices_id[id+1]:
                edges.append((vertices_id[id]-1, vertices_id[id+1]-1))

        if vertices_id[-1] != vertices_id[0]:
            edges.append((vertices_id[-1]-1, vertices_id[0]-1))
        
        return edges


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

    _integer_regex = re.compile("([-+]?[0-9]+)")
    _float_regex = re.compile("([+-]?([0-9]+([.][0-9]*)?|[.][0-9]+))")
    _non_comment_regex = re.compile("^(.*)#")
    _modelbegin_regex = re.compile("^ *(modelbegin).*$", re.IGNORECASE)
    _modelend_regex = re.compile("^ *(modelend).*$", re.IGNORECASE)
    _clumpbegin_regex = re.compile("^ *(clumpbegin).*$", re.IGNORECASE)
    _clumpend_regex = re.compile("^ *(clumpend).*$", re.IGNORECASE)
    _protobegin_regex = re.compile("^ *(protobegin) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _protoinstance_regex = re.compile("^ *(protoinstance) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _protoend_regex = re.compile("^ *(protoend).*$", re.IGNORECASE)
    _vertex_regex = re.compile("^ *(vertex|vertexext)(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){3}) *(uv(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){2}))?.*$", re.IGNORECASE)
    _polygon_regex = re.compile("^ *(polygon|polygonext)( +[0-9]+)(( +[0-9]+)+) ?.*$", re.IGNORECASE)
    _quad_regex = re.compile("^ *(quad|quadext)(( +([0-9]+)){4}).*$", re.IGNORECASE)
    _triangle_regex = re.compile("^ *(triangle|triangleext)(( +([0-9]+)){3}).*$", re.IGNORECASE)
    _texture_regex = re.compile("^ *(texture) +([A-Za-z0-9_\\-]+).*$", re.IGNORECASE)
    _color_regex = re.compile("^ *(color)(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){3}).*$", re.IGNORECASE)
    _opacity_regex = re.compile("^ *(opacity)( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)).*$", re.IGNORECASE)
    _transform_regex = re.compile("^ *(transform)(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){16}).*$", re.IGNORECASE)
    _scale_regex = re.compile("^ *(scale)(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){3}).*$", re.IGNORECASE)
    _rotate_regex = re.compile("^ *(rotate)(( +[-+]?[0-9]*){4})$", re.IGNORECASE)
    _surface_regex = re.compile("^ *(surface)(( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)){3}).*$", re.IGNORECASE)
    _ambient_regex = re.compile("^ *(ambient)( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)).*$", re.IGNORECASE)
    _diffuse_regex = re.compile("^ *(diffuse)( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)).*$", re.IGNORECASE)
    _specular_regex = re.compile("^ *(specular)( +[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)).*$", re.IGNORECASE)

    # End regex list

    def __init__(self, uri, default_surface=(0.0, 0.0, 0.0)):
        
        self._rwx_clump_stack = []
        self._rwx_proto_dict = {}
        self._current_scope = None

        with fileinput.input(files=(uri,)) as f:
            res = None
            for line in f:

                old_line = ""
 
                #strip comment away
                res = self._non_comment_regex.match(line)
                if res:
                    old_line = line
                    line = res.group(1)
                
                res = self._modelbegin_regex.match(line)
                if res:
                    self._rwx_clump_stack.append(RwxObject())
                    self._current_scope = self._rwx_clump_stack[-1]
                    self._current_scope.state.surface = default_surface
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
                    vprops = [ float(x[0]) for x in self._float_regex.findall(res.group(2)) ]
                    if res.group(7):
                        vprops.extend([ float(x[0]) for x in self._float_regex.findall(res.group(7)) ])
                        self._current_scope.vertices.append(RwxVertex(vprops[0], vprops[1], vprops[2], u = vprops[3], v = vprops[4]))
                    else: self._current_scope.vertices.append(RwxVertex(vprops[0], vprops[1], vprops[2]))
                    continue

                res = self._color_regex.match(line)
                if res:
                    cprops = [ float(x[0]) for x in self._float_regex.findall(res.group(2)) ]
                    if len(cprops) == 3:
                        self._current_scope.state.color = tuple(cprops)
                    continue

                res = self._opacity_regex.match(line)
                if res:
                    self._current_scope.state.opacity = float(res.group(2))
                    continue

                res = self._transform_regex.match(line)
                if res:
                    tprops = [ float(x[0]) for x in self._float_regex.findall(res.group(2)) ]
                    if len(tprops) == 16: self._current_scope.state.transform = mu.Matrix(list(zip(*[iter(tprops)]*4))).transposed()
                    continue

                res = self._rotate_regex.match(line)
                if res:
                    rprops = [ int(x) for x in self._integer_regex.findall(res.group(2)) ]
                    if len(rprops) == 4:
                        if rprops[0]:
                            self._current_scope.state.transform =\
                            mu.Matrix.Rotation(radians(-rprops[3]), 4, 'X') * self._current_scope.state.transform
                        if rprops[1]:
                            self._current_scope.state.transform =\
                            mu.Matrix.Rotation(radians(-rprops[3]), 4, 'Y') * self._current_scope.state.transform
                        if rprops[2]:
                            self._current_scope.state.transform =\
                            mu.Matrix.Rotation(radians(-rprops[3]), 4, 'Z') * self._current_scope.state.transform
                    continue
                    
                res = self._scale_regex.match(line)
                if res:
                    sprops = [ float(x[0]) for x in self._float_regex.findall(res.group(2)) ]
                    if len(sprops) == 3:
                        self._current_scope.state.transform =\
                        mu.Matrix.Scale(sprops[0], 4, (1.0, 0.0, 0.0)) *\
                        mu.Matrix.Scale(sprops[1], 4, (0.0, 1.0, 0.0)) *\
                        mu.Matrix.Scale(sprops[2], 4, (0.0, 0.0, 1.0)) * self._current_scope.state.transform
                    continue

                res = self._surface_regex.match(line)
                if res:
                    sprops = [ float(x[0]) for x in self._float_regex.findall(res.group(2)) ]
                    if len(sprops) == 3:
                        self._current_scope.state.surface = tuple(sprops)
                    continue

                res = self._ambient_regex.match(line)
                if res:
                    surf = self._current_scope.state.surface
                    self._current_scope.state.surface = (float(res.group(2)), surf[1], surf[2])
                    continue

                res = self._diffuse_regex.match(line)
                if res:
                    surf = self._current_scope.state.surface
                    self._current_scope.state.surface = (surf[0], float(res.group(2)), surf[2])
                    continue

                res = self._specular_regex.match(line)
                if res:
                    surf = self._current_scope.state.surface
                    self._current_scope.state.surface = (surf[0], surf[1], float(res.group(2)))
                    continue


    def __call__(self):
        return self._rwx_clump_stack[0]


def add_attr_recursive(clump, name):

    attr = []
    attr.extend(getattr(clump, name))
    
    for c in clump.clumps:
        attr.extend(add_attr_recursive(c, name))

    return attr


def add_vertices_recursive(clump, transform = mu.Matrix.Identity(4)):

    vertices = []
    transform = transform * clump.state.transform
    for v in clump.verts:
        vert = transform * mu.Vector([v[0], v[1], v[2], 1])
        vertices.append((vert[0], vert[1], vert[2]))

    for c in clump.clumps:
        vertices.extend(add_vertices_recursive(c))

    return vertices


def add_faces_recursive(clump, offset=0):

    faces = []
    polys = []
    tmp_faces = clump.faces
    tmp_polys = clump.polys
    
    for tmp_face in tmp_faces:
        faces.append((tmp_face[0]+offset, tmp_face[1]+offset, tmp_face[2]+offset))
        
    for tmp_poly in tmp_polys:
        polys.append([(edge[0]+offset, edge[1]+offset) for edge in tmp_poly])
    
    offset += len(clump.verts)
    
    for c in clump.clumps:
        tmp_faces, tmp_polys, offset = add_faces_recursive(c, offset)
        faces.extend(tmp_faces)
        polys.extend(tmp_polys)

    return faces, polys, offset


def make_materials_recursive(ob, clump, folder, extension = "jpg"):

    for shape in clump.shapes:
        # Get material

        mat_sign = shape.state.mat_signature
        
        mat = bpy.data.materials.get(mat_sign)
        
        if mat is None:
            # create material
            mat = bpy.data.materials.new(name=mat_sign)
            
            if shape.state.texture and folder is not None:
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

            mat.alpha = shape.state.opacity
            mat.diffuse_color = shape.state.color
            mat.specular_color = (1.0, 1.0, 1.0)
            mat.ambient = shape.state.surface[0]
            mat.diffuse_intensity = shape.state.surface[1]
            mat.specular_intensity = shape.state.surface[2]
            mat.alpha = shape.state.opacity

            ob.data.materials.append(mat)

        else:
            if mat_sign not in ob.data.materials.keys():
                ob.data.materials.append(mat)
        
    for sub_clump in clump.clumps:
        make_materials_recursive(ob, sub_clump, folder, extension)

if in_blender:
        
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

        default_ambient= FloatProperty(
            name="Default Ambient",
            description="Default ambient light intensity for materials",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            precision=3)

        default_diffuse= FloatProperty(
            name="Default Diffuse",
            description="Default diffuse light intensity for materials",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            precision=3)

        default_specular= FloatProperty(
            name="Default Specular",
            description="Default specular light intensity for materials",
            default=0.0,
            min=0.0,
            max=1.0,
            step=0.1,
            precision=3)

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
                    self.report({'WARNING'}, "No texture directory specified and could not guess any, no texture will be used.")
                    texturepath = None

            elif not os.path.isdir(texturepath):
                self.report({'WARNING'}, "Texture directory specified is not a directory, no texture will be used.")
                texturepath = None
                
            try:
                parser = RwxParser(filepath, default_surface=(self.default_ambient,\
                                                              self.default_diffuse,\
                                                              self.default_specular))
                rwx_object = parser()
            except Exception as exc:
                print_exc(exc)
                self.report({'ERROR'}, "Could not parse input file (either not a proper .rwx or a bug on my part).")
                return {'CANCELLED'}

            if len(rwx_object.clumps) == 0:
                self.report({'ERROR'}, "No clump registered after parsing the input file (likely not a proper .rwx).")
                return {'CANCELLED'}

            verts = add_vertices_recursive(rwx_object.clumps[0])
            faces, polys, offset = add_faces_recursive(rwx_object.clumps[0])
            faces_state = add_attr_recursive(rwx_object.clumps[0], "faces_state")
            polys_state = add_attr_recursive(rwx_object.clumps[0], "polys_state")
            faces_uv = add_attr_recursive(rwx_object.clumps[0], "faces_uv")
            verts_uv = add_attr_recursive(rwx_object.clumps[0], "verts_uv")
            
            mesh = bpy.data.meshes.new('Mesh')
            ob = bpy.data.objects.new('Object', mesh)

            make_materials_recursive(ob, rwx_object.clumps[0], texturepath)

            # Create mesh from given verts, edges, faces. Either edges or
            # faces should be [], or you ask for problems
            mesh.from_pydata(verts, [], faces)
            bm = bmesh.new()
            bm.from_mesh(mesh)

            uv_layer = bm.loops.layers.uv.verify()
            bm.faces.layers.tex.verify()  # currently blender needs both layers.
                
            if uv_layer is None:
                uv_layer = bm.loops.layers.uv.new()

            # Adjust materials and UVs for faces
            for i, f in enumerate(bm.faces):
                f.material_index = ob.data.materials.keys().index(faces_state[i].mat_signature)
                for j, l in enumerate(f.loops):
                    uv = faces_uv[i][j]
                    if uv[0] is not None and uv[1] is not None:
                        l[uv_layer].uv = uv

            bm.verts.ensure_lookup_table()

            # Now we need to fill polygon with triangles (make faces)
            
            for i, poly in enumerate(polys):
                bm_edges = []

                first_vert = bm.verts.new((verts[poly[0][0]][0], verts[poly[0][0]][1], verts[poly[0][0]][2]))
                prev_vert = first_vert
                verts_uv.append((verts_uv[poly[0][0]][0], verts_uv[poly[0][0]][1]))
                bm.verts.ensure_lookup_table()
                
                for edge in poly[:-1]:
                    new_vert = bm.verts.new((verts[edge[1]][0], verts[edge[1]][1], verts[edge[1]][2]))
                    verts_uv.append((verts_uv[edge[1]][0], verts_uv[edge[1]][1]))
                    bm_edge = bm.edges.new((prev_vert, new_vert))
                    bm.verts.ensure_lookup_table()
                    prev_vert = new_vert
                    bm.edges.ensure_lookup_table()
                    bm_edges.append(bm_edge)

                bm_edge = bm.edges.new((prev_vert, first_vert))
                bm.edges.ensure_lookup_table()
                bm_edges.append(bm_edge)
                
                geom = edgeloop_fill(bm, edges=bm_edges)["faces"]

                # adjust materials and UVs for polygons
                
                for f in geom:
                    if isinstance(f, bmesh.types.BMFace):
                        f.material_index = ob.data.materials.keys().index(polys_state[i].mat_signature)
                        for l in f.loops:
                            uv = verts_uv[l.vert.index]
                            if uv[0] is not None and uv[1] is not None:
                                l[uv_layer].uv = uv

            bm.faces.ensure_lookup_table()
                    
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
    op = self.layout.operator(Rwx2BlenderOperator.bl_idname, text="Active Worlds RenderWare script (.rwx)")

def register():
    bpy.utils.register_class(Rwx2BlenderOperator)
    bpy.types.INFO_MT_file_import.append(import_menu_func_rwx)

def unregister():
    bpy.utils.unregister_class(Rwx2BlenderOperator)
    bpy.types.INFO_MT_file_import.remove(import_menu_func_rwx)

if __name__ == "__main__":
    
    if in_blender: register()

