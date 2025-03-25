import numpy as np
import matplotlib.pyplot as plt

def circle_segment_endpoints(center, radius, n):
    x0, y0 = center
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = [(x0 + radius * np.cos(theta), y0 + radius * np.sin(theta)) for theta in angles]

    # Connect each point to the next
    segments = []
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]  # wrap around to close the circle
        segments.append((p1, p2))

    return segments

# Example usage
center = (0, 0)
radius = 5
n = 8  # Number of segments
apothem = radius / np.cos(np.pi / n)
segments = circle_segment_endpoints(center, apothem, n)

# Print segment endpoints
for i, ((x1, y1), (x2, y2)) in enumerate(segments):
    print(f"Segment {i+1}: ({x1:.3f}, {y1:.3f}) -> ({x2:.3f}, {y2:.3f})")

# Plot the approximation
plt.figure(figsize=(6,6))
for (x1, y1), (x2, y2) in segments:
    plt.plot([x1, x2], [y1, y2], 'b')

# Plot actual circle for reference
theta = np.linspace(0, 2 * np.pi, 300)
plt.plot(center[0] + radius * np.cos(theta), center[1] + radius * np.sin(theta), 'r--', label='Actual Circle')

plt.gca().set_aspect('equal')
plt.title(f"Circle Approximation with {n} Segments")
plt.grid(True)
plt.legend()
plt.show()
