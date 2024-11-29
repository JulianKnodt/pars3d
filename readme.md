# pars3d

A standalone 3D parsing library.
Currently supports the following formats:


#### Primary Support
- .obj
- .glb (with `feature = "gltf"`, binary only)
- .fbx (import only for now)


##### Secondary Support
- .off
- .stl
- .ply

| |obj|glb|fbx|off|stl|ply|
|-|  -|  -|  -|  -|  -|  -|
|import|Y|Y|Y|Y|Y|Y|
|export|Y|Y|In progress|Y|N|Y|
|Unified Repr|Y|Y|Y|N|N|N|

Unified Repr indicates that there is a single struct which can be used to interchange between
each format. That is the main difference between primary and secondary support. Each secondary
support format has to be treated uniquely.

# Design:

Each file format has its own `struct` which represents the data supported by that specific file
type. These structs are then unifiable into a single generic mesh representation.

# Visualization

In addition to parsing 3D file formats, there is minimal support for visualization with vertex
colors. This can be used to output PLY files with per edge colors.

# Notes:

Inspired by [Assimp](https://github.com/assimp/)!


![VRML](https://web.archive.org/web/20000929035521/http://www.geocities.com:80/SiliconValley/4944/VRML.gif)
