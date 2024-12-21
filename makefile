# temporary test makefile for testing that FBX exports fields corretly
test_fbx:
	cargo run --release -- cube.fbx tmp.fbx
	cargo run --release -- tmp.fbx tmp2.fbx

parse_fbx:
	cargo run --release -- cube.fbx tmp.fbx
