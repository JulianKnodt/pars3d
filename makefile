# temporary test makefile for testing that FBX exports fields corretly
cube_fbx_to_obj:
	cargo run --release -- cube.fbx tmp.obj
parse_fbx_to_obj:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.obj

parse_fbx:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.fbx

test_fbx:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.fbx
	cargo run --release -- tmp.fbx tmp2.fbx

