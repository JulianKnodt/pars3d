use super::F;
use std::array::from_fn;
use std::io::{self, BufRead, BufReader, Read};

use std::path::Path;

#[derive(Debug, Clone)]
pub struct MSH {
    pub nodes: Vec<[F; 3]>,
    pub elements: Vec<Element>,
    pub element_fields: Vec<Field>,
}

#[derive(Debug, Clone, Copy)]
pub enum Element {
    Tri([usize; 3]),
    Quad([usize; 3]),
    Tet([usize; 4]),
    Hex([usize; 8]),
}

impl Element {
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        match self {
            Self::Tri(i) => i.as_mut_slice(),
            Self::Tet(i) => i.as_mut_slice(),
            Self::Hex(i) => i.as_mut_slice(),
            Self::Quad(i) => i.as_mut_slice(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: String,
    pub components_per_entry: u32,
    pub data: Vec<f64>,
}

pub fn read_from_file(p: impl AsRef<Path>) -> io::Result<MSH> {
    read(std::fs::File::open(p)?)
}

pub fn read(reader: impl Read) -> io::Result<MSH> {
    buf_reader(BufReader::new(reader))
}

pub fn buf_reader(mut reader: impl BufRead) -> io::Result<MSH> {
    let mut buf = String::new();
    reader.read_line(&mut buf)?;
    assert_eq!(buf, "$MeshFormat\n");
    buf.clear();
    reader.read_line(&mut buf)?;
    let mut e = buf.split_whitespace();

    let _version = e.next().expect("Missing version").parse::<f64>().unwrap();

    let ty = e.next().expect("Missing type").parse::<u32>().unwrap();
    let is_binary = ty == 1;

    let data_size = e.next().expect("Missing data size").parse::<u32>().unwrap();
    assert_eq!(data_size, 8);

    if is_binary {
        let mut buf = [0; 4];
        reader.read_exact(&mut buf)?;
        let one = u32::from_ne_bytes(buf);
        assert_eq!(one, 1, "Different endianness than expected");
    }
    buf.clear();

    reader.read_line(&mut buf)?;
    assert_eq!(buf, "$EndMeshFormat\n");

    buf.clear();
    let mut byte_buf: Vec<u8> = vec![];

    // are these vertices?
    let mut nodes = vec![];

    let mut elements = vec![];

    let mut element_fields = vec![];

    while let Ok(c) = reader.read_line(&mut buf) {
        if c == 0 {
            break;
        }
        match buf.as_str() {
            "$Nodes\n" => {
                buf.clear();
                reader.read_line(&mut buf)?;
                let num_nodes = buf.trim().parse::<usize>().unwrap();
                nodes.resize(num_nodes, [0.; 3]);
                if is_binary {
                    let stride = (4 + 3 * data_size) as usize;
                    let num_bytes = stride * num_nodes;
                    byte_buf.resize(num_bytes as usize, 0);
                    reader.read_exact(&mut byte_buf)?;
                    for i in 0..num_nodes {
                        let s = &byte_buf[i * stride..(i + 1) * stride];
                        let idx = i32::from_ne_bytes(from_fn(|i| s[i])) - 1;
                        assert!(idx >= 0);
                        let offset = 4;
                        let x = f64::from_ne_bytes(from_fn(|i| s[i + offset]));
                        let offset = 12;
                        let y = f64::from_ne_bytes(from_fn(|i| s[i + offset]));
                        let offset = 20;
                        let z = f64::from_ne_bytes(from_fn(|i| s[i + offset]));
                        nodes[idx as usize] = [x as F, y as F, z as F];
                    }
                } else {
                    todo!("Implement ascii version of nodes");
                }

                buf.clear();
                reader.read_line(&mut buf)?;
                assert_eq!(buf, "$EndNodes\n");
            }
            "$Elements\n" => {
                buf.clear();
                reader.read_line(&mut buf)?;
                let num_elem = buf.trim().parse::<usize>().unwrap();
                assert!(is_binary, "TODO implement ascii");

                let mut elem_read = 0;
                while elem_read < num_elem {
                    let mut int_buf = [0; 4];
                    reader.read_exact(&mut int_buf)?;
                    let elem_type = i32::from_ne_bytes(int_buf);
                    reader.read_exact(&mut int_buf)?;
                    let elem_type = ElemKind::from_num(elem_type);

                    let num_elems = i32::from_ne_bytes(int_buf);
                    elements.resize(num_elems as usize, Element::Tri([0; 3]));
                    reader.read_exact(&mut int_buf)?;
                    let num_tags = i32::from_ne_bytes(int_buf);

                    for _ in 0..num_elems {
                        reader.read_exact(&mut int_buf)?;
                        let elem_idx = u32::from_ne_bytes(int_buf);
                        let elem_idx = elem_idx - 1;
                        for _ in 0..num_tags {
                            reader.read_exact(&mut int_buf)?;
                        }
                        let mut curr = match elem_type {
                            ElemKind::Tri => Element::Tri([0; 3]),
                            ElemKind::Quad => Element::Quad([0; _]),
                            ElemKind::Tet => Element::Tet([0; _]),
                            ElemKind::Hex => Element::Hex([0; _]),
                        };
                        let cs = curr.as_mut_slice();
                        for j in 0..elem_type.num_nodes() {
                            reader.read_exact(&mut int_buf)?;
                            let node_idx = i32::from_ne_bytes(int_buf) - 1;
                            cs[j] = node_idx as usize;
                        }
                        elements[elem_idx as usize] = curr;
                    }
                    elem_read += num_elems as usize;
                }

                buf.clear();
                reader.read_line(&mut buf)?;
                assert_eq!(buf, "$EndElements\n");
            }
            "$ElementData\n" => {
                buf.clear();
                reader.read_line(&mut buf)?;
                let num_string_tags = buf.trim().parse::<usize>().unwrap();
                let mut string_tags = vec![String::new(); num_string_tags];

                for i in 0..num_string_tags {
                    buf.clear();
                    reader.read_line(&mut buf)?;
                    string_tags[i].push_str(buf.trim());
                }

                buf.clear();
                reader.read_line(&mut buf)?;
                let num_real_tags = buf.trim().parse::<usize>().unwrap();
                let mut real_tags = vec![0.; num_real_tags];

                for i in 0..num_real_tags {
                    buf.clear();
                    reader.read_line(&mut buf)?;
                    real_tags[i] = buf.trim().parse::<f64>().unwrap();
                }

                buf.clear();
                reader.read_line(&mut buf)?;
                let num_int_tags = buf.trim().parse::<usize>().unwrap();
                let mut int_tags = vec![0; num_int_tags];

                for i in 0..num_int_tags {
                    buf.clear();
                    reader.read_line(&mut buf)?;
                    int_tags[i] = buf.trim().parse::<i32>().unwrap();
                }
                assert!(num_string_tags > 0);
                assert!(num_int_tags > 2);

                let field_name = &string_tags[0];
                let num_components = int_tags[1] as u32;
                let num_entries = int_tags[2] as usize;

                let mut vals = vec![];
                assert!(is_binary, "TODO implement ascii");
                let stride = (num_components * data_size + 4) as usize;
                let num_bytes = stride * num_entries;

                let mut data = vec![0; num_bytes as usize];
                reader.read_exact(&mut data)?;

                for i in 0..num_entries {
                    let s = &data[i * stride..(i + 1) * stride];
                    let idx = i32::from_ne_bytes(from_fn(|i| s[i])) - 1;
                    assert!(idx >= 0);
                    let mut offset = 4;
                    for _ in 0..num_components {
                        let val = f64::from_ne_bytes(from_fn(|i| s[i + offset]));
                        vals.push(val);
                        offset += 8;
                    }
                }

                buf.clear();
                reader.read_line(&mut buf)?;
                assert_eq!(buf, "$EndElementData\n");

                element_fields.push(Field {
                    name: field_name.clone(),
                    components_per_entry: num_components,
                    data: vals,
                });
            }
            _l => eprintln!("Skipping line {_l:?}"),
        }
        buf.clear();
    }

    Ok(MSH {
        nodes,
        elements,
        element_fields,
    })
}

#[derive(Clone, Copy, Debug)]
enum ElemKind {
    Tri,
    Quad,
    Tet,
    Hex,
}

impl ElemKind {
    fn from_num(i: i32) -> Self {
        match i {
            2 => Self::Tri,
            3 => Self::Quad,
            4 => Self::Tet,
            5 => Self::Hex,
            _ => panic!(),
        }
    }
    const fn num_nodes(self) -> usize {
        match self {
            Self::Tri => 3,
            Self::Quad => 4,
            Self::Tet => 4,
            Self::Hex => 8,
        }
    }
}

#[test]
fn test_load_msh() {
    if let Err(e) = read_from_file("output_dwn.msh") {
        panic!("{e:?}");
    };
}
