from functions import *

#parameters

input_folder = "postures"
my_file = "BabyHighRes.ply"
theta = 30.
phi = 30.

ref_vertex = "ref_vert.txt" #temporary method to ensure the right direction of normal vectors

print(read_vertex_ref(ref_vertex))

#---------------

#take ply and get vertex and plane coordinates in rigth way fr the model
planes_vector, vertices_vector, colors_vector, normals_array = get_from_input(input_folder + "/" + my_file)
# WAIT!: normal vectors are not orientated!

#compute normal vectors for each surface
normal_vector = get_normal(planes_vector, vertices_vector)

#rotate reference frame of angle theta and phi (2 rotations)
vec_rotated = rotate_ref_frame(vertices_vector, theta, phi)

#orthograpical projection into plane ortogonal to the current direction
vec_proj, vec_proj_reduce = proj(vec_rotated)

#compute barycenter of each triangle
vec_of_bar = triangle_barycenter(planes_vector, vertices_vector)

#ret = compute_light_vector(planes_vector, normals_array, sun_vector(theta, phi))
#print_ply(ret, vec_proj, colors_vector)


reduce_dimension_by_shadow(planes_vector, vec_proj_reduce)

#print the result
#print_ply(planes_vector, vec_proj, colors_vector) #original one

