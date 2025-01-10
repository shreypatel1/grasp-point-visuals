import numpy as np

import math_models
import open3d as o3d
from sklearn.decomposition import PCA


# get XYZ points from txt file seperated by comma
file_path = 'data/all_points4.txt'

points = []
with open(file_path, 'r') as f:
    for line in f:
        x, y, z = line.split(',')
        points.append([float(x), float(y), float(z)])

points = np.array(points)

# Create an Open3D point cloud object
pcd = o3d.geometry.PointCloud()

if len(points) > 0:
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=120, std_ratio=0.5,
                                              print_progress=True)
    # get the updated set of points as a numpy array
    points = np.asarray(pcd.points)
    line_set = o3d.geometry.LineSet()

    # RANSAC Cylinder Fitting
    distance_threshold = 0.01
    ransac_n = 500  # Minimum number of points to fit a cylinder model
    num_iterations = 1000  # Number of RANSAC iterations

    # Generate plane samples and their normal vectors to calculate the cylinder axis
    mmdl = math_models.MathModels()
    plane_samples = mmdl.generate_plane_samples(points, num_samples=ransac_n, max_distance=distance_threshold)

    # Add lines representing the plane normals
    line_pts = []
    lineset_lines = []
    displacements = []
    normals = []

    for sample in plane_samples:
        p1, p2, p3 = sample
        center = (p1 + p2 + p3) / 3
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)

        # Ensure the normal vector points away from the origin
        if np.dot(center, normal) < 0:
            normal = -normal

        displacements.append(center)
        normals.append([normal * 0.05, center])

        line_pts.append(center)
        line_pts.append(center + normal * 0.03)  # Scale the normal for visualization
        lineset_lines.append([len(line_pts) - 2, len(line_pts) - 1])

    line_pts = np.array(line_pts)
    lineset_lines = np.array(lineset_lines, dtype=np.int32)
    normals = np.array(normals)

    # Add the points and lines to the line_set object
    line_set.points = o3d.utility.Vector3dVector(line_pts)
    line_set.lines = o3d.utility.Vector2iVector(lineset_lines)

    # Calculate the cylinder axis line using the plane normals and least squares fitting
    pca = PCA(n_components=1)
    pca.fit(displacements)
    initial_axis = pca.components_[0]
    axis_point = np.mean(displacements, axis=0)

    axis = initial_axis
    for _ in range(num_iterations):  # Iterate to refine the direction
        # Calculate projections of normals onto the current axis direction
        projection_magnitudes = np.dot(normals, axis)

        # Calculate orthogonal density by minimizing projections along the axis direction
        orthogonal_density = np.sum(np.abs(projection_magnitudes))

        # Slightly adjust the axis direction to maximize orthogonal density
        rotation_matrix = np.eye(3) + 0.1 * np.random.randn(3, 3)  # Small random rotation
        rotated_direction = rotation_matrix @ axis
        rotated_direction /= np.linalg.norm(rotated_direction)

        new_projection_magnitudes = np.dot(normals, rotated_direction)
        new_orthogonal_density = np.sum(np.abs(new_projection_magnitudes))

        # Update if the new direction yields a better orthogonal alignment
        if new_orthogonal_density < orthogonal_density:
            axis_direction = rotated_direction
            orthogonal_density = new_orthogonal_density

    line_ptsa = np.array(
        [axis_point, axis_point + axis * 0.1])  # Extend line in the direction of axis
    lineseta_lines = np.array([[0, 1]], dtype=np.int32)  # Only one line segment

    print(f"Axis: {axis}")
    # print(f"Axis Point: {axis_point}")

    line_seta = o3d.geometry.LineSet()
    line_seta.points = o3d.utility.Vector3dVector(line_ptsa)
    line_seta.lines = o3d.utility.Vector2iVector(lineseta_lines)

    # Project all the points onto an orthogonal plane to the axis
    projection, basis_x, basis_y = mmdl.project_points_to_plane(points, axis_point, axis)

    # Estimate the circle center and radius
    cylinder_center, cylinder_radius = mmdl.estimate_circle_center_radius(projection, basis_x, basis_y, axis_point)

    print(f"Cylinder Center: {cylinder_center}")
    print(f"Cylinder Radius: {cylinder_radius}")

    # Create a cylinder object
    # cylinder_mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=0.15)
    # cylinder_mesh.paint_uniform_color([0.1, 0.1, 0.7])

    # Translate the cylinder to the center of the cylinder
    # translated_mesh = cylinder_mesh.translate(cylinder_center, relative=False)

    # Rotate the cylinder to align with the axis of the cylinder
    # cylinder_mesh.rotate(o3d.geometry.get_rotation_matrix_from_xyz(axis), center=cylinder_center)

    # Display everything
    # cylinder_mesh.compute_vertex_normals()
    # line_seta
    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd, line_set])