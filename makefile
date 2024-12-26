# temporary test makefile for testing that FBX exports fields corretly
cube_fbx_roundtrip:
	cargo run --release --bin fbx_roundtrip -- cube.fbx tmp.fbx
	cargo run --release --bin fbx_roundtrip -- tmp.fbx tmp2.fbx

parse_fbx_to_obj:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.obj

parse_fbx:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.fbx

test_fbx:
	cargo run --release -- cube.fbx tmp.fbx
	cargo run --release -- tmp.fbx tmp2.fbx

