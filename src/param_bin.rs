use std::io::{Read, Write};

use crate::*;

pub fn export_to_file(params: &[Param], path: &str) {
    let f = std::fs::File::create(path).unwrap();
    let mut writer = std::io::BufWriter::new(f);
    for param in params {
        let tensor = param.get_tensor();
        writer
            .write_all(&(tensor.ndim() as u32).to_le_bytes())
            .unwrap();
        for s in tensor.shape() {
            writer.write_all(&(*s as u32).to_le_bytes()).unwrap();
        }
        for x in tensor.iter() {
            writer.write_all(&x.to_le_bytes()).unwrap();
        }
    }
}

pub fn import_from_file(params: &mut [Param], path: &str) {
    let f = std::fs::File::open(path).unwrap();
    let mut reader = std::io::BufReader::new(f);
    let mut buf = [0u8; 4];
    for param in params {
        reader.read(&mut buf).unwrap();
        let ndim = u32::from_le_bytes(buf) as usize;
        let mut shape = Vec::new();
        for _ in 0..ndim {
            reader.read(&mut buf).unwrap();
            shape.push(u32::from_le_bytes(buf) as usize);
        }
        let mut data = Vec::with_capacity(shape.iter().product());
        for _ in 0..shape.iter().product() {
            reader.read(&mut buf).unwrap();
            data.push(f32::from_le_bytes(buf));
        }
        param.set(NDArray::from_shape_vec(shape, data).unwrap());
    }
}

pub fn params_summary(params: &[Param]) {
    let mut total_size = 0;
    for (i, param) in params.iter().enumerate() {
        let shape = param.get_tensor().shape().to_vec();
        let size = shape.iter().product::<usize>();
        total_size += size;
        println!("{:>2}{:>8} {:?}", i, size, shape);
    }
    println!("floats: {} ({} bytes)", total_size, total_size * 4);
}

#[test]
fn test() {
    use crate::optimizers::*;
    let ndarrays = vec![
        NDArray::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        NDArray::from_shape_vec(vec![3, 4], (0..12).map(|x| x as f32).collect()).unwrap(),
    ];
    let mut params = vec![
        Param::new(ndarrays[0].clone(), Fixed),
        Param::new(ndarrays[1].clone(), Fixed),
    ];
    let path = "/tmp/tensorflake_param_bin_test.bin";
    export_to_file(&params, path);
    import_from_file(&mut params, path);
    assert_eq!(&*params[0].get_tensor(), &ndarrays[0]);
    assert_eq!(&*params[1].get_tensor(), &ndarrays[1]);
}
