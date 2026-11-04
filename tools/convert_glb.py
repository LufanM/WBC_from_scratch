"""
convert_glb.py — bake the RealSense d435 GLB into an OBJ MuJoCo can load.

MuJoCo has no glTF support: ``mjuu_extToContentType`` (upstream
src/user/user_util.cc) maps only .stl/.obj/.ply/.msh, and the mesh decoder
plugin set is obj/stl/usd only. This holds as of 3.11. So the camera body
shipped as ``meshes/realsense/d435.glb`` has to be transcoded offline; the
URDF's <visual> block for it is dropped during conversion otherwise.

The GLB is a 5-part textured scene at ~231k triangles. We concatenate the
parts into one mesh, weld duplicate vertices and drop UVs/materials: the
camera is a small decorative body on the wrist, so vertex geometry alone is
what MuJoCo needs.

Run:
    python3 tools/convert_glb.py       # writes meshes/realsense/d435.obj
"""
from __future__ import annotations

import os
import sys

# This script lives in tools/; the robot assets are under the repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "assets", "galaxea_r1pro_standalone")
SRC = os.path.join(PKG, "meshes", "realsense", "d435.glb")
DST = os.path.join(PKG, "meshes", "realsense", "d435.obj")


def convert(src=SRC, dst=DST):
    import numpy as np
    import trimesh

    scene = trimesh.load(src, force="scene")
    mesh = scene.to_mesh()          # flatten the 5 parts, applying node transforms
    mesh.merge_vertices()           # the parts share seams; weld them
    mesh.visual = trimesh.visual.ColorVisuals()   # drop textures/UVs

    # MuJoCo needs a consistent outward winding to shade the mesh sensibly.
    mesh.fix_normals()

    with open(dst, "w", encoding="utf-8") as f:
        f.write("# baked from d435.glb by convert_glb.py\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.6g} {v[1]:.6g} {v[2]:.6g}\n")
        for face in mesh.faces + 1:  # OBJ indices are 1-based
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"[glb] wrote {dst}")
    print(f"[glb] verts={len(mesh.vertices)} faces={len(mesh.faces)} "
          f"extents={np.round(mesh.extents, 5)}")
    return dst


if __name__ == "__main__":
    convert(*(sys.argv[1:3] or []))
