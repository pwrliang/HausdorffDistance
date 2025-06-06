import glob
import json

import open3d as o3d
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt, image as mpimg
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd
import os


def compute_hausdorff(pc1, pc2):
    tree2 = cKDTree(pc2)

    dist1, indices1 = tree2.query(pc1, k=1)

    max_index = np.argmax(dist1)
    return dist1[max_index], pc1[max_index], pc2[indices1[max_index]]


def draw_model():
    # Load two PLY files
    model1 = o3d.io.read_triangle_mesh("/Users/liang/ModelNet40/chair/train/chair_0653.off")
    model2 = o3d.io.read_triangle_mesh("/Users/liang/ModelNet40/chair/train/chair_0653.off")
    bbox = model1.get_axis_aligned_bounding_box()

    # (Optional) Translate the second point cloud so they don’t overlap
    model2.translate((25, 0, 0))  # Shift along x-axis

    lineset1 = o3d.geometry.LineSet.create_from_triangle_mesh(model1)
    lineset1.paint_uniform_color([1, 0, 0])  # Optional: black lines

    lineset2 = o3d.geometry.LineSet.create_from_triangle_mesh(model2)
    lineset2.paint_uniform_color([0, 0, 1])  # Optional: black lines

    points1 = np.asarray(model1.vertices)
    points2 = np.asarray(model2.vertices)

    print("Model 1 has", len(points1), "points")
    print("Model 2 has", len(points2), "points")

    hd, point1, point2 = compute_hausdorff(points1, points2)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([point1, point2])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[0, 0, 0]])

    geometries = [lineset1, lineset2, line_set]

    # Define rotation: angle (radians), axis = Y
    angle_deg = -25
    angle_rad = np.radians(angle_deg)

    # Rotation matrix for Y-axis
    R = o3d.geometry.get_rotation_matrix_from_axis_angle([angle_rad, np.radians(8), 0])

    # Apply rotation to all geometries
    for geom in geometries:
        geom.rotate(R, center=(0, 0, 0))  # Rotate around origin

    # Headless rendering and screenshot capture
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # headless
    for geom in geometries:
        vis.add_geometry(geom)
    ctr = vis.get_view_control()

    params = ctr.convert_to_pinhole_camera_parameters()

    # Extract and print camera info
    extrinsic = params.extrinsic
    intrinsic = params.intrinsic

    # Compute eye, lookat, up from extrinsic matrix
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    eye = -R.T @ t  # camera position
    lookat = eye + R.T @ np.array([0, 0, 1])  # looking toward +Z in camera space
    up = R.T @ np.array([0, -1, 0])  # up is -Y in camera space

    print("Eye (camera position):", eye)
    print("Lookat (target point):", lookat)
    print("Up vector:", up)

    ctr.set_front([10, -30, 50])  # viewing direction
    ctr.set_lookat([10, 25, 0])  # target (center of scene)
    ctr.set_up([.0, 1, 0.0])  # up vector
    ctr.set_zoom(0.7)  # zoom level

    opt = vis.get_render_option()
    opt.line_width = 8

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image("rendered.png")  # saves PNG
    vis.destroy_window()
    from PIL import Image, ImageChops

    def trim_white_border(image_path, output_path):
        img = Image.open(image_path).convert("RGB")
        bg = Image.new("RGB", img.size, (255, 255, 255))  # white background
        diff = ImageChops.difference(img, bg)
        bbox = diff.getbbox()
        if bbox:
            trimmed = img.crop(bbox)
            trimmed.save(output_path)
            print(f"Cropped image saved to: {output_path}")
        else:
            print("No non-white content found.")

    trim_white_border("rendered.png", "rendered.png")


def draw_time():
    def load_df(files, cache_file):
        # Otherwise, load from JSON files and serialize
        json_records = []
        for file in files:
            with open(file) as f:
                json_records.append(json.load(f))

        df = pd.json_normalize(json_records)

        # Serialize to pickle
        df.to_pickle(cache_file)
        print("Loaded DataFrame from JSON and saved to cache.")
        return df

    all_files = glob.glob(os.path.join("logs/intro", '*.json'))
    df = load_df(all_files, "intro.pkl")

    df_eb = df[df['Running.Repeats'].apply(lambda x: x[0]['Algorithm'] == 'Early Break')]
    df_eb = df_eb.sort_values(by="Input.Translate")
    df_eb['Early Break'] = df['Running.AvgTime']
    df_eb['Input.Translate'] = df_eb['Input.Translate'] * 100
    df_eb = df_eb[['Early Break', 'Input.Translate']]
    df_eb.set_index('Input.Translate', inplace=True)


    df_nn = df[df['Running.Repeats'].apply(lambda x: x[0]['Algorithm'] == 'Nearest Neighbor Search')]
    df_nn = df_nn.sort_values(by="Input.Translate")
    df_nn['Input.Translate'] = df_nn['Input.Translate'] * 100
    df_nn['NN Search'] = df['Running.AvgTime']
    df_nn = df_nn[['NN Search', 'Input.Translate']]
    df_nn.set_index('Input.Translate', inplace=True)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    img = mpimg.imread('rendered.png')  # or use PIL.Image.open if preferred
    axes[0].imshow(img)
    axes[0].axis('off')  # Hide axes
    axes[0].set_title('(a) Two same chair models and its HD', y=-0.25)
    axes[0].text(0.1, 0.08, "Hausdorff\nDistance", transform=axes[0].transAxes, fontsize=10, )
    # axes[1].set_xticks([1, 2, 4, 8, 16, 32])
    df_eb.plot(kind='line', marker='o', ax=axes[1])
    df_nn.plot(kind='line', marker='o', ax=axes[1], ls='dashed')

    # Set x-axis ticks to be the same as the 'X_Coordinate' data points
    # axes[1].set_xticks(df['Input.Translate'])

    axes[1].margins(x=0.05, y=0.25)
    axes[1].set_xlabel('Translate x-axis proportionally to model size (%)')
    axes[1].set_ylabel('Running Time (ms)')
    axes[1].set_title("(b) Running time by moving the blue model", y=-0.25)
    axes[1].legend(loc='upper left', ncol=2, handletextpad=0.3,
                   fontsize=11, borderaxespad=0.2, frameon=False)
    #
    axes[1].set_xscale('log', base=2)
    def format_power_of_2(value, pos):
        return f'{int(value)}'

    # Apply formatter to x-axis
    axes[1].xaxis.set_major_formatter(ticker.FuncFormatter(format_power_of_2))

    plt.tight_layout()
    fig.savefig('intro.pdf', format='pdf', bbox_inches='tight')
    plt.show()


# draw_model()
draw_time()
