import sys

from scipy.spatial import ConvexHull
import numpy as np
import cvxopt


class Mesh(object):
    def __init__(self, vertices, faces):
        """Simple mesh object.

        Args:
            vertices : ndarray, shape (n_vertices, 3)
                Array of vertices.
            faces : ndarray, shape (n_faces, 3)
                Array of faces.
        """
        self._vertices = vertices
        self._faces = faces
        self._barycenter = vertices.mean(axis=0)
        self._normals = self.compute_normals()
        self._edges = None
        self._vertex_face_neighbors = None

    @property
    def vertices(self):
        return self._vertices
    
    @property
    def faces(self):
        return self._faces

    @property
    def edges(self):
        if self._edges is None:
            self.compute_edges()
        return self._edges
    
    @property
    def normals(self):
        return self._normals
    
    @property
    def barycenter(self):
        return self._barycenter

    def compute_normals(self):
        normals = np.zeros(self._faces.shape)
        for face_idx, face in enumerate(self._faces):
            normals[face_idx] = self.compute_face_normal(face)
        return normals

    def compute_face_normal(self, face):
        p0 = self._vertices[face[0]]
        p1 = self._vertices[face[1]]
        p2 = self._vertices[face[2]]
        n = np.cross(p1 - p0, p2 - p0)

        # Check that normal is oriented away from the barycenter
        n = n if (np.dot(n, p0 - self.barycenter) > 0) else -n
        
        # Normalize
        n = n / np.linalg.norm(n)
        return n

    def compute_edges(self):
        e0 = self._faces[..., :2]
        e1 = self._faces[..., 1:]
        e2 = self._faces[..., [2, 0]]
        edges = np.concatenate([e0, e1, e2], axis=0)
        np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)
        self._edges = edges

    def vertex_face_neighbors(self, idx):
        if self._vertex_face_neighbors is None:
            self.compute_vertex_face_neighbors()

        return self._vertex_face_neighbors[idx]

    def compute_vertex_face_neighbors(self):
        vertex_face_neighbors = [None]*self._vertices.shape[0]
        for f_idx, f in enumerate(self._faces):
            for v_idx in f:
                if not vertex_face_neighbors[v_idx]: vertex_face_neighbors[v_idx] = []
                vertex_face_neighbors[v_idx].append(f_idx)
        self._vertex_face_neighbors = vertex_face_neighbors

    def collapse_edge(self, edge_idx, new_position=None):
        edge = self.edges[edge_idx]
        n_vertices = self._vertices.shape[0]
        v0 = edge[0]
        v1 = edge[1]
        neigh_face_v0 = set(self.vertex_face_neighbors(v0))
        neigh_face_v1 = set(self.vertex_face_neighbors(v1))

        # Assert edge is ordered by index
        if v0 > v1: v0, v1 = v1, v0

        # Update vertex position with fallback to mid-point
        self._vertices[v0] = new_position if new_position is not None else (self._vertices[v0] + self._vertices[v1]) / 2

        # Remove redundant vertex
        self._vertices = np.delete(self._vertices, v1, axis=0)

        # Update vertex indices in faces
        vertex_index_lookup = np.array([v0] * n_vertices)
        vertex_index_lookup[np.arange(n_vertices) != v1] = np.arange(n_vertices - 1)
        self._faces = vertex_index_lookup[self._faces]

        # Update normals of impacted faces
        collapse_faces = neigh_face_v0.intersection(neigh_face_v1)
        update_faces = neigh_face_v0.union(neigh_face_v1) - collapse_faces
        for face_idx in update_faces:
            self._normals[face_idx] = self.compute_face_normal(self._faces[face_idx])

        # Remove collapsed faces
        self._faces = np.delete(self._faces, list(collapse_faces), axis=0)
        self._normals = np.delete(self._normals, list(collapse_faces), axis=0)

        # Remove edges and face adjacency so they are recomputed
        self._edges = None
        self._vertex_face_neighbors = None


def edge_contract_smallest_added_volume(mesh):
    """
    Edge contraction by smallest added volume.
    Code adopted from Jinchao Tan et al. [2016],
    edited to require fewer dependencies.

    Args:
        mesh : Mesh, mesh to be simplified. 
    """
    vertices = mesh.vertices
    faces = mesh.faces
    edges = mesh.edges
    
    temp_list1 = []
    count = 0

    for edge in edges:
        vertex1 = edge[0]
        vertex2 = edge[1]
        face_index1 = mesh.vertex_face_neighbors(vertex1)
        face_index2 = mesh.vertex_face_neighbors(vertex2)

        face_index = list(set(face_index1) | set(face_index2))
        old_face_list = []
        
        #### now find a point, so that for each face in related_faces will create a positive volume tetrahedron using this point.
        ### minimize c*x. w.r.t. A*x<=b
        c = np.zeros(3)
        A = []
        b = []

        for index in face_index:
            face = faces[index]
            p0 = vertices[face[0]]
            p1 = vertices[face[1]]
            p2 = vertices[face[2]]
            old_face_list.append(np.asarray([p0, p1, p2]))
            n = mesh.normals[index] 
            A.append(n)
            b.append(np.dot(n, p0))
            c += n
                
        ########### now use cvxopt.solvers.lp solver  
        A = -np.asarray(A).astype(np.double)
        b = -np.asarray(b).astype(np.double)
        
        c = np.asarray(c).astype(np.double)
        cvxopt.solvers.options['show_progress'] = False
        cvxopt.solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')
        res = cvxopt.solvers.lp( cvxopt.matrix(c), cvxopt.matrix(A), cvxopt.matrix(b), solver='glpk' )

        if res['status'] == 'optimal':
                
            newpoint = np.asarray(res['x']).astype(np.double).squeeze()
        
            ####### manually compute volume as priority,so no relation with objective function
            tetra_volume_list = []
            for each_face in old_face_list:
                tetra_volume_list.append(compute_tetrahedron_volume(each_face, newpoint))
            volume = np.asarray(tetra_volume_list).sum()
            
            temp_list1.append((count, volume, newpoint))
        else:
            # maximum value for float
            temp_list1.append((count, sys.float_info.max, vertices[vertex1]))
        count += 1
              
    if len(temp_list1) == 0:
        print('No edge could be contracted because of solver failure')
    else:
        min_tuple = min(temp_list1, key=lambda x: x[1])
        edge_index = min_tuple[0]
        new_vertex_position = min_tuple[2]
        mesh.collapse_edge(edge_index, new_vertex_position)

    return mesh

        
def compute_tetrahedron_volume(face, point):
    n = np.cross(face[1] - face[0], face[2] - face[0])
    return abs(np.dot(n, point - face[0])) # Should actually divide by 6, but not necessary to find minimum
