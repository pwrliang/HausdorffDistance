import open3d as o3d

# Load two PLY files
model1 = o3d.io.read_point_cloud("/Users/liang/dragon_recon/dragon_vrip_res3.ply")
model2 = o3d.io.read_point_cloud("/Users/liang/happy_recon/happy_vrip_res3.ply")

# Apply colors
model1.paint_uniform_color([1, 0, 0])  # Red
model2.paint_uniform_color([0, 0, 1])  # Blue

# (Optional) Translate the second point cloud so they don’t overlap
# model2.translate((0.2, 0, 0))  # Shift along x-axis

model1 = model1.voxel_down_sample(voxel_size=0.005)
model2 = model2.voxel_down_sample(voxel_size=0.005)

print("Model 1 has", len(model1.points), "points")
print("Model 2 has", len(model2.points), "points")


# Visualize both together
o3d.visualization.draw_geometries([model1, model2])

# model1.compute_vertex_normals()
# model2.compute_vertex_normals()
# o3d.visualization.draw_geometries([model1, model2], mesh_show_wireframe=True)