from functions import *

#parameter

file = "BabyHighRes.ply"
theta = 0.
phi = 0. 
#---------------
planes_vector, vertices_vector, colors_vector = get_from_input(file)

normal_vector = get_normal(planes_vector, vertices_vector)

vec_rotated = rotate_ref_frame(vertices_vector, theta, phi)

vec_proj, vec_proj_reduce = proj(vec_rotated)

print_ply(planes_vector, vec_proj, colors_vector)

