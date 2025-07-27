import open3d
import numpy as np


def get_sphere(position, radius=1.0, color=(0.1, 0.1, 0.7)):
    mesh_sphere: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.paint_uniform_color(color)
    mesh_sphere = mesh_sphere.translate(position, relative=False)
    return mesh_sphere

def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def get_cylinder(start_point, end_point, radius=0.3, color=(0.1, 0.9, 0.1)):
    center = (start_point + end_point) / 2
    height = np.linalg.norm(start_point - end_point)
    mesh_cylinder: open3d.geometry.TriangleMesh = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    mesh_cylinder.paint_uniform_color(color)

    rot_vec = end_point - start_point
    rot_vec = rot_vec / np.linalg.norm(rot_vec)
    rot_0 = np.array([0, 0, 1])
    rot_mat = rotation_matrix_from_vectors(rot_0, rot_vec)

    rotation_param = rot_mat
    mesh_cylinder = mesh_cylinder.rotate(rotation_param)
    mesh_cylinder = mesh_cylinder.translate(center, relative=False)
    return mesh_cylinder