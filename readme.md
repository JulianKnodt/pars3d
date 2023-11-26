# pars3d

A simple standalone 3D parsing library.
Currently supports the following formats:

- .obj
- .off
- .stl

# Design:

Unlike Assimp, each file has a different structure. This means that each format will not be
burdened by the structure of others, but adds additional complexity.
