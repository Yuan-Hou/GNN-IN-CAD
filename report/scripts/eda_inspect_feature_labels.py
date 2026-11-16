from __future__ import annotations
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

"""Interactive inspection of supplied machining-feature labels."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import _load_sample
import numpy as np
import plotly.graph_objects as go

def triangulate_uv_face(grid):
    points = grid[..., :3]
    valid = grid[..., 6] > .5
    triangles = []
    for u in range(grid.shape[0] - 1):
        for v in range(grid.shape[1] - 1):
            for corners in [((u, v), (u+1, v), (u+1, v+1)),
                            ((u, v), (u+1, v+1), (u, v+1))]:
                if all(valid[a, b] for a, b in corners):
                    triangle = np.array([points[a, b] for a, b in corners])
                    if np.linalg.norm(np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])) > 1e-12:
                        triangles.append(triangle)
    return np.asarray(triangles).reshape(-1, 3, 3)


def mesh_trace(triangles, name, color, opacity):
    vertices = triangles.reshape(-1, 3)
    indices = np.arange(len(vertices)).reshape(-1, 3)
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=indices[:, 0], j=indices[:, 1], k=indices[:, 2],
                     name=name, color=color, opacity=opacity, flatshading=True,
                     hovertemplate=name + "<extra></extra>",
                     lighting=dict(ambient=.55, diffuse=.75, specular=.15))


def plot_label_examples(label):
    """Show two solids bearing a selected original supplied class name."""
    from data import _load_splits
    from data import LABEL_MAPPING
    if label not in LABEL_MAPPING:
        raise ValueError(f'Unknown supplied class {label!r}. Choose from {sorted(LABEL_MAPPING)}')
    selected = []
    for sample_id in _load_splits()['train']:
        sample = _load_sample(sample_id)
        if any(face['segment']['name'] == label for face in sample['faces']):
            selected.append(sample)
        if len(selected) == 2:
            break
    for sample in selected:
        fig = go.Figure()
        for face_index, grid in enumerate(sample['graph'].ndata['x'].numpy()):
            triangles = triangulate_uv_face(grid)
            if not len(triangles):
                continue
            face_label = sample['faces'][face_index]['segment']['name']
            fig.add_trace(mesh_trace(triangles, f'Face {face_index}: {face_label}',
                                     '#ce353b' if face_label == label else '#a5afb9', 1.0))
        fig.update_layout(
            title=f"{sample['id']} — supplied {label.replace('_', ' ')} labels in red",
            height=550, margin=dict(l=0, r=0, b=0, t=50), showlegend=False,
            scene=dict(aspectmode='data', dragmode='orbit',
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                camera=dict(eye=dict(x=-1.7, y=-1.7, z=1.3))))
        fig.show(renderer='plotly_mimetype', config=dict(scrollZoom=True, displaylogo=False))
