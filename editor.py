import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import Scale, HORIZONTAL

# Load landmark points
landmarks = np.load("landmarks.npy")  # shape: (N, 3)
spheres = []

# Visualize as small spheres in Open3D
def make_spheres(landmarks):
    return [o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(p) for p in landmarks]

# Callback: update landmark + Open3D viewer
def update_landmark(index, axis, value):
    landmarks[index][axis] = float(value)
    spheres[index] = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(landmarks[index])
    refresh_viewer()

def refresh_viewer():
    vis.clear_geometries()
    for s in spheres:
        vis.add_geometry(s)

# Initialize Open3D window
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Face Landmark Editor")
spheres = make_spheres(landmarks)
for s in spheres:
    vis.add_geometry(s)

# Tkinter UI
root = tk.Tk()
root.title("Landmark Control Panel")

for idx in range(len(landmarks)):
    for axis, name in enumerate(['X', 'Y', 'Z']):
        scale = Scale(root, from_=-0.5, to=0.5, resolution=0.01,
                      orient=HORIZONTAL,
                      label=f'Pt{idx} {name}',
                      command=lambda val, i=idx, a=axis: update_landmark(i, a, val))
        scale.set(landmarks[idx][axis])
        scale.pack()

def loop():
    vis.poll_events()
    vis.update_renderer()
    root.after(100, loop)

def update_open3d():
    vis.poll_events()
    vis.update_renderer()
    root.after(50, update_open3d)



update_open3d()
root.after(100, loop)
root.mainloop()
