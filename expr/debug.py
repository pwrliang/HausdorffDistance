import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Step 1: Define the triangles
h = 10
triangles = [
    [(-86.211708, 32.866695, h), (-86.211708, 32.866695, 0.0), (-86.418579, 33.073563, 0.0)],
    [(-86.418579, 33.073563, h), (-86.418579, 33.073563, 0.0), (-86.625450, 32.866695, 0.0)],
    [(-86.625450, 32.866695, h), (-86.625450, 32.866695, 0.0), (-86.418579, 32.659828, 0.0)],
    [(-86.418579, 32.659828, h), (-86.418579, 32.659828, 0.0), (-86.211708, 32.866695, 0.0)],

    [(-86.046127, 32.838188, h), (-86.046127, 32.838188, 0.0), (-86.252998, 33.045055, 0.0)],
    [(-86.252998, 33.045055, h), (-86.252998, 33.045055, 0.0), (-86.459869, 32.838188, 0.0)],
    [(-86.459869, 32.838188, h), (-86.459869, 32.838188, 0.0), (-86.252998, 32.631321, 0.0)],
    [(-86.252998, 32.631321, h), (-86.252998, 32.631321, 0.0), (-86.046127, 32.838188, 0.0)],

]



# Step 2: Define the ray
ray_origin = np.array([-86.413124, 32.707394, 5])
ray_direction = np.array([0, 1, 0])  # Positive Y direction

# Step 3: Ray-triangle intersection using Möller–Trumbore algorithm
def ray_intersects_triangle(ray_origin, ray_vector, triangle):
    EPSILON = 1e-6
    vertex0, vertex1, vertex2 = [np.array(v) for v in triangle]
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0
    h = np.cross(ray_vector, edge2)
    a = np.dot(edge1, h)
    if -EPSILON < a < EPSILON:
        return None  # Ray is parallel
    f = 1.0 / a
    s = ray_origin - vertex0
    u = f * np.dot(s, h)
    if u < 0.0 or u > 1.0:
        return None
    q = np.cross(s, edge1)
    v = f * np.dot(ray_vector, q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = f * np.dot(edge2, q)
    if t > EPSILON:
        return ray_origin + ray_vector * t
    return None

# Step 4: Check for intersections
intersections = []
for idx, tri in enumerate(triangles):
    intersection = ray_intersects_triangle(ray_origin, ray_direction, tri)
    if intersection is not None:
        intersections.append((idx, intersection))

# Step 5: Visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw triangles and labels
for idx, tri in enumerate(triangles):
    poly = Poly3DCollection([tri], alpha=0.7, edgecolor='k')
    ax.add_collection3d(poly)
    centroid = tuple(sum(coords) / 3 for coords in zip(*tri))
    ax.text(*centroid, str(idx), color='red', fontsize=12, weight='bold')

# Draw ray origin
ax.scatter(*ray_origin, color='blue', s=50, label='Ray Origin')

# Draw ray line
ray_end = ray_origin + ray_direction * 1.0  # Adjust length as needed
ax.plot(
    [ray_origin[0], ray_end[0]],
    [ray_origin[1], ray_end[1]],
    [ray_origin[2], ray_end[2]],
    color='blue',
    linewidth=2,
    label='Ray Direction'
)

# Draw intersection points
for idx, point in intersections:
    ax.scatter(*point, color='green', s=50, label=f'Intersection @ Triangle {idx}')
    ax.text(*point, f'Hit {idx}', color='green', fontsize=10)

# Set labels and view
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Elevation')
ax.set_title("Ray-Triangle Intersection Visualization")
ax.auto_scale_xyz([-86.7, -86.2], [32.6, 33.1], [0, h*2])
ax.view_init(elev=20, azim=45)
ax.legend()

plt.tight_layout()
plt.show()
