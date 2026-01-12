# pars3d

![shapes](assets/shapes.gif)

A 3D parsing library with minimal dependencies.

Currently supports the following formats:

### Basic Usage:

I/O for multiple formats:
```rust
use pars3d;

let scene: pars3d::Scene = load("my_mesh.fbx").expect("Failed to load");
...
pars3d::save("new_mesh.fbx", scene).expect("Failed to save");
```

I/O for specific formats (varies per format):
```rust
use pars3d::fbx;

let fbx_scene = fbx::parser::load("my_mesh.fbx").expect("Failed to load");
...
let mut out = std::fs::File::create("new_mesh.fbx");
fbx::export::export_fbx(fbx_scene, std::io::BufWriter::new(out)).expect("Failed to save");
```


#### Primary Support
- .obj
- .glb (with `feature = "gltf"`, binary only)
- .fbx (experimental export, binary only)
- .ply (ascii + binary, vertex colors, normals, UV, ![NEW](https://web.archive.org/web/20091027143217if_/http://geocities.com/dent30cmu/icon/new.gif) Gaussian Splats)


##### Secondary Support
- .stl
- .off


| |obj|glb|fbx|off|stl|ply|
|-|  -|  -|  -|  -|  -|  -|
|import|Y|Y|Y (no anim)|Y|Y|Y|
|export|Y|Y|In progress|Y|Y|Y|
|Unified Repr|Y|Y|Y|Y|Y|Y|
|Allocations|Few, Large|Few, Large|Many, Small|?|?|?|

Unified Repr indicates that there is a single struct which can be used to interchange between
each format.

Secondary formats are considered less important for maintenance.

If you want another format supported, please feel free to file an issue and I'll do my best to
add it.

### Contributing is welcome!

File an issue, submit a PR, anything would be fantastic.

# Visualization

In addition to parsing 3D file formats, there is minimal support for visualization with vertex
colors. This can be used to output PLY files with per edge colors.

### Design:

Each file format has its own `struct` which represents the data supported by that specific file
type. These structs are then unifiable into a single generic mesh representation.


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

### Shortcomings:

This library may allocate more than necessary when deserializing. For many applications,
deserialization is the final step, but if I/O is done in a long-living process this may be
slower than necessary.

For FBX parsing, my philosophy is if there's something I didn't expect it should crash.
Immediately. This is great for figuring out exactly what is wrong, but if your input FBX file
doesn't fit what is expected it will not work at all. Better than silently being wrong though.

If you do not want this behavior, disable default features, specifically `strict_fbx`, which
will stop crashes on unknown fields.

Furthermore, it doesn't currently do much for changing materials between formats, which I'll add
in at some point.

#### Why does this exist?

I started this library primarily for parsing 3D models for geometry processing. Initially, it
started just for parsing OBJ files (one of the main formats used in research), but found that
a lot of meshes I wanted to parse where in other formats, specifically FBX. Originally, I could
not be arsed to make an FBX parser, and since Sketchfab also provided a GLTF export, I
downloaded the models as GLTF files. For most models this is fine, but I realized thatthe GLTF
format has a number of shortcomings for geometry processing, so I bit the bullet and started an
FBX importer/exporter. I also added PLY because I wanted to be able to visualize values on
edges/faces/vertices, and PLY has great support for vertex colors.

Another benefit of this library is that the original polygon format will be preserved. For quad
and triangle meshes, all faces will be stack allocated, meaning that traversal is efficient, and
there are only a handful of larger allocations. If your mesh consists of many polygonal faces
with more than 4 edges, they will be heap allocated. I haven't seen a mesh with mostly polygonal
faces, so I think the trade-off is well worth it.

##### Comparison to Alternatives

Assimp - C, I/O for FBX is a bit buggy, more feature complete than this library

fbxcel - Rust, requires deep knowledge of the FBX file format, deprecated
