use std::io::Write;

use crate::*;

pub fn export_to_file(params: &[Variable], path: &str) {
    let f = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(f);
    for param in params {
        for x in param.iter() {
            writer.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

// pub fn import_from_file(params: &mut [Variable], path: &str) {
//     let f = std::fs::File::open(path).unwrap();
//     let mut reader = std::io::BufReader::new(f);
//     for param in params {
//         for x in param.iter_mut() {
//             x.from_le_bytes(&mut reader).unwrap();
//         }
//     }
// }