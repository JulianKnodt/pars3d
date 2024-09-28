# pars3d

A standalone 3D parsing library.
Currently supports the following formats:

- .obj
- .off
- .stl
- .ply
- .glb/.gltf (with `feature = "gltf"`, and only some support)

# Design:

Each file format has its own `struct` which represents the data supported by that specific file
type. These structs are then unifiable into a single generic mesh representation.

# Visualization

In addition to parsing 3D file formats, there is minimal support for visualization with vertex
colors. This can be used to output PLY files with per edge colors.
