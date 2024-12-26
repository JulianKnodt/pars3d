#![feature(os_str_display)]

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let mut fbx_files = vec![];
    for f in std::env::args().skip(1) {
        if !f.ends_with(".fbx") {
            eprintln!("Found file which was not an FBX file, exiting");
            return Ok(());
        }
        fbx_files.push(f);
    }
    if fbx_files.is_empty() {
        eprintln!("Usage: <bin> <...fbx files>");
        return Ok(());
    }

    for filename in fbx_files {
        let f = File::open(&filename)?;
        use pars3d::fbx::parser::{parse_tokens, tokenize_binary};
        let tokens = tokenize_binary(BufReader::new(f)).expect("Failed to tokenize FBX");
        let kvs = parse_tokens(tokens.into_iter());
        let mut out_name = PathBuf::from(filename);
        assert!(out_name.set_extension("dot"));
        let vis = File::create(out_name.file_name().unwrap())?;
        kvs.to_graphviz(BufWriter::new(vis))?;
        println!("[INFO]: Saved {}", out_name.file_name().unwrap().display());
    }

    Ok(())
}
