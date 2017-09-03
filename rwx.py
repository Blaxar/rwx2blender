import re
import fileinput
import numpy as np
from copy import copy, deepcopy
import os
import numpy as np

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

    
    def _instanciate_proto(self, name):
        pass
        
    
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
                    v_id = [ int(x) for x in self._integer_regex.findall(res.group(3)) ]
                    self._current_scope.shapes.append(RwxPolygon(v_id,\
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

    def __call__(self):
        return self._rwx_clump_stack[0]
