import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from math import *

np.set_printoptions(suppress=True)

def get_from_input(file):
    """
    file must be a char
    """
    mesh = o3d.io.read_triangle_mesh(file)
    pcd = o3d.io.read_point_cloud(file, format='xyzrgb')

    colors_array = np.asarray(pcd.colors)
    print("Shape of colors is: ", np.shape(colors_array))

    triangles_array = np.asarray(mesh.triangles)
    print("Shape of triangles is: ", np.shape(triangles_array))

    vertices_array = np.asarray(mesh.vertices)
    print("Shape of vertices is: ", np.shape(vertices_array))

    return triangles_array, vertices_array, colors_array

def print_ply(plane, vertex, color):
    """
    len_vert = len(vertex[0])
    
    if(len_vert == 2):
        vert_y = []
        for i in range(len(vertex)):
            vert_y.append(0.)
    else:
        vert_y = vertex[:,2]
     """  
     
    vert_z = vertex[:,2]
    vert_x = vertex[:,0]
    vert_y = vertex[:,1]

    face_x = plane[:,0]
    face_y = plane[:,1]
    face_z = plane[:,2]

    fig = go.Figure(data=[
          go.Mesh3d(
            x=vert_x,
            y=vert_y,
            z=vert_z,
            #colorbar_title='intensity',
            #colorscale=[[0, 'gold'],
            #            [0.5, 'mediumturquoise'],
            #            [1, 'magenta']],
            # Intensity of each vertex, which will be interpolated and color-coded
            vertexcolor = color,
            # i, j and k give the vertices of triangles

            # here we represent the 4 triangles of the tetrahedron surface
            i=face_z,
            j=face_x,
            k=face_y,
            name='y',
            showscale=True)])

    fig.show()  
    return 0


def concatenate(single_plane, vertex):
    """
    given a plane and vector of vertices
    this function result is coordinate 
    in the space of that plane
    """
    
    #note: the PLY file start from 1 and not from 0

    comp_x = vertex[single_plane[0] - 1]
    comp_y = vertex[single_plane[1] - 1]
    comp_z = vertex[single_plane[2] - 1]

    return comp_x, comp_y, comp_z

def get_normal(planes, vertex):
    tot_l = len(planes)
    data_vec = []
    for i in range(tot_l):
        comp_x, comp_y, comp_z = concatenate(planes[i], vertex)
        data_vec.append(np.cross(comp_y - comp_x, comp_z - comp_y))

    return data_vec

def rotate_ref_frame(vec, theta, phi):
    
    theta = theta*pi/180.
    phi = phi*pi/180.
    vec_pro = vec

    #rotation around z axis
    rot_phi = ([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])

    #rotation around y axis
    rot_theta = ([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

    vec_first_rotation = np.dot(vec, rot_phi) #product of matrix Nx3 x 3x3 = Nx3
    vec_result = np.dot(vec_first_rotation, rot_theta) #product of matrix Nx3 x 3x3 = Nx3

    return vec_result

def proj(vec):

    my_vec = np.ndarray(shape=(len(vec),3))
    my_vec_reduce = np.ndarray(shape=(len(vec),2)) # necessary for the transformation

    #we must take the x and z component
    my_vec[:, 0] = vec[:, 0]
    my_vec[:, 1] = vec[:, 2]

    my_vec_reduce[:, 0] = vec[:, 0]
    my_vec_reduce[:, 1] = vec[:, 2]

    for i in range(len(vec)):
        my_vec[:,2] = 0.

    return my_vec, my_vec_reduce

def segno(a, b, p):
    discr = 0.00001
    c = p[0]*(a[1] - b[1]) - \
        p[1]*(a[0] - b[0]) + \
        a[0]*b[1] - a[1]*b[0]
    if(np.abs(c)<0):
        return 0 #allieate with the line
    elif(c>0):
        return 1 #one hemiplane
    else:
        return -1 #other hemiplane

def in_ex_point(a, b, c, p):

    s1 = segno(a, b, c)*segno(a, b, p)
    s2 = segno(b, c, a)*segno(b, c, p)
    s3 = segno(c, a, b)*segno(c, a, p)

    if(s1<0. and s2<0. and s3<0.):
        return False
    elif(s1>0. and s2>0. and s3>0.):
        return True #the point is inside
    else:
        return False #the point is on the border (very unprobably)

#the idea now is check which point is included in a certain plain (for 2d vector of course)
