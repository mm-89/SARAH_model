import open3d as o3d
import numpy as np
import plotly.graph_objects as go
from math import *

np.set_printoptions(suppress=True)

def read_vertex_ref(file_to_read):
    a1 = []
    with open(file_to_read) as f:
        for line in f:
            dat = line.split()
            a1.append(int(dat[0]))
    return int(a1[0])

def ref_vector(ref_point, vertex):
    val = vertex[ref_point]
    val[1] += 2 #translate y value of two
    return val - ref_point

def establish_normal(normal_vector, ref_vector, ref_vertex):
    pass

def get_from_input(file_to_read):
    """
    file must be a char
    """
    mesh = o3d.io.read_triangle_mesh(file_to_read)
    pcd = o3d.io.read_point_cloud(file_to_read, format='xyzrgb')

    """
    pcd.estimate_normals()
    pcd.normalize_normals()
    normals_array = np.asarray(pcd.normals) #it's empty
    """

    #o3d.visualization.draw_geometries([pcd])

    if(mesh.is_orientable()):
        print("This Mesh is Orientable")
        mesh.orient_triangles()

#    help(mesh)
#    help(o3d.io)

    colors_array = np.asarray(pcd.colors)
    print("Shape of colors is: ", np.shape(colors_array))

    triangles_array = np.asarray(mesh.triangles)
    print("Shape of triangles is: ", np.shape(triangles_array))

    vertices_array = np.asarray(mesh.vertices)
    print("Shape of vertices is: ", np.shape(vertices_array))

    mesh.compute_triangle_normals(normalized=True) # directions are uncorrect

    if(mesh.has_triangle_normals()):
        print("This Mesh has triangle normals")
        mesh.normalize_normals()
    else:
        print("This Mesh doentn't have triangle normals")

    normals_array = np.asarray(mesh.triangle_normals)

    return triangles_array, vertices_array, colors_array, normals_array

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

def triangle_barycenter(plane, vertex):
    """
    this function is needed to understand
    which from two triangles are in front
    of each other. It will be necessary 
    to define a method more logic
    """
    vec_of_bar = np.empty(shape=(len(plane),3))
    for i in range(len(vec_of_bar)):
        comp_x, comp_y, comp_z = concatenate(plane[i], vertex)
        vec_of_bar[i] = np.array([np.sum(comp_x)/3., np.sum(comp_y)/3., np.sum(comp_z)/3.])

    return vec_of_bar

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
        return False #the point is outside
    elif(s1>0. and s2>0. and s3>0.):
        return True #the point is inside
    else:
        return False #the point is on the border (very unprobably)

#another algorithm, faster than the previous
def in_ex_point_alt(a, b, c, p):

    d = (b[0] - a[0])*(c[1] - a[1]) - (b[1] - a[1])*(c[0] - a[0])
    a_cmp = ((p[0] - a[0])*(c[1] - a[1]) - (p[1] - a[1])*(c[0] - a[0]))/d
    b_cmp = ((a[0] - p[0])*(b[1] - a[1]) - (a[1] - p[1])*(b[0] - a[0]))/d
    if(a_cmp > 0 and b_cmp > 0 and a_cmp+b_cmp < 1):
        return True
    else:
        return False

def sun_vector(theta, phi):
    return np.array([np.sin(theta)*np.cos(theta),
            np.sin(theta)*np.sin(phi),
            np.cos(theta)])

#the idea now is check which point is included in a certain plain (for 2d vector of course)

def compute_light_vector(plane_vec, normal_vec, sun_vec):
    #just to check
    if(len(plane_vec) != len(normal_vec)):
            print("Something gone wrong")
    new_vec = []
    i = 0
    for nv in normal_vec:
        if(np.dot(nv, sun_vec) < 0.):
            new_vec.append(plane_vec[i])
            i += 1
    return np.array(new_vec)




def reduce_dimension_by_shadow(plane, vertex_2D):
    """
    remeber: vertex_2D is reduce plane vector,
    with three components of couple of numbersi

    It is the vertex that has been projected, not 
    the plane. It is always the same
    """

    #to be continued...
    new_plane = []
    num = len(plane)
    i = 1;
    for sp in plane:
        print("Compute: ", i , "/", num)
        comp_x, comp_y, comp_z = concatenate(sp, vertex_2D)
        i += 1
        for vt in vertex_2D:
            if(in_ex_point_alt(comp_x, comp_y, comp_z, vt)):

    return True

        
