import open3d as o3d

mesh = o3d.io.read_triangle_mesh("/Users/liang/3dmodels/thai_statuette.ply")
# e.g. simplify to 50k triangles
mesh_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=9996)
o3d.io.write_triangle_mesh("/Users/liang/3dmodels/thai_statuette_decimated.ply", mesh_dec)