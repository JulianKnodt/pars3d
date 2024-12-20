# pars3d

A 3D parsing library with minimal dependencie.
Currently supports the following formats:


#### Primary Support
- .obj
- .glb (with `feature = "gltf"`, binary only)
- .fbx (experimental export, binary only)
- .ply


##### Secondary Support
- .off
- .stl

| |obj|glb|fbx|off|stl|ply|
|-|  -|  -|  -|  -|  -|  -|
|import|Y|Y|Y|Y|Y|Y|
|export|Y|Y|In progress|Y|N|Y|
|Unified Repr|Y|Y|Y|N|N|Y|

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

Unlike Assimp, each file format has its own representation, which can then be converted into a
unified representation:

- On Import `File -> Tokenization/Parser -> Per Format Struct -> Unified`
- On Export `Unified -> Per Format Struct -> Parser/Tokenization -> File`

This is in contrast to Assimp, which instead has:

- On Import `File -> Tokenization/Parser -> Unified`
- On Export `Unified -> Parser/Tokenization -> File`

What I've noticed is that because there is no per format struct, it is difficult to maintain
metadata which varies between each representation. It is also more difficult to distinguish bugs
in the exporter itself and misrepresentations in the struct. By separating out each format, it
makes it more clear what can be exported. This is also because it is much easier to make an
importer and exporter for a single file format. This was inspired by intermediate
representations (IRs) in compilers which allow for performing certain tasks more easily.

The one downside is that now there is more code to maintain. I believe this trade-off is worth it
though.

![VRML](https://web.archive.org/web/20000929035521/http://www.geocities.com:80/SiliconValley/4944/VRML.gif)
