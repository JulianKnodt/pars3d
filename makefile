# temporary test makefile for testing that FBX exports fields corretly
parse_fbx:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.fbx

test_fbx:
	cargo run --release -- Spartan_Sketchfab.fbx tmp.fbx
	cargo run --release -- tmp.fbx tmp2.fbx

