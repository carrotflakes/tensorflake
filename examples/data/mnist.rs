#![allow(dead_code)]

pub struct Mnist {
    pub train_images: Vec<u8>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<u8>,
    pub test_labels: Vec<u8>,
}

impl Mnist {
    pub fn load(dir: &str) -> Mnist {
        use std::io::Read;
        let f = std::fs::File::open(format!("{}/{}", dir, "train-images-idx3-ubyte")).unwrap();
        let mut r = std::io::BufReader::new(f);
        let mut header = [0u8; 16];
        r.read(&mut header).unwrap();
        let is: Vec<_> = header
            .chunks(4)
            .map(|x| i32::from_be_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(&is, &[2051, 60000, 28, 28]);
        let mut train_images = vec![0u8; 60000 * 28 * 28];
        r.read_exact(&mut train_images).unwrap();

        let f = std::fs::File::open(format!("{}/{}", dir, "train-labels-idx1-ubyte")).unwrap();
        let mut r = std::io::BufReader::new(f);
        let mut header = [0u8; 8];
        r.read(&mut header).unwrap();
        let is: Vec<_> = header
            .chunks(4)
            .map(|x| i32::from_be_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(&is, &[2049, 60000]);
        let mut train_labels = vec![0u8; 60000];
        r.read_exact(&mut train_labels).unwrap();

        let f = std::fs::File::open(format!("{}/{}", dir, "t10k-images-idx3-ubyte")).unwrap();
        let mut r = std::io::BufReader::new(f);
        let mut header = [0u8; 16];
        r.read(&mut header).unwrap();
        let is: Vec<_> = header
            .chunks(4)
            .map(|x| i32::from_be_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(&is, &[2051, 10000, 28, 28]);
        let mut test_images = vec![0u8; 10000 * 28 * 28];
        r.read_exact(&mut test_images).unwrap();

        let f = std::fs::File::open(format!("{}/{}", dir, "t10k-labels-idx1-ubyte")).unwrap();
        let mut r = std::io::BufReader::new(f);
        let mut header = [0u8; 8];
        r.read(&mut header).unwrap();
        let is: Vec<_> = header
            .chunks(4)
            .map(|x| i32::from_be_bytes(x.try_into().unwrap()))
            .collect();
        assert_eq!(&is, &[2049, 10000]);
        let mut test_labels = vec![0u8; 10000];
        r.read_exact(&mut test_labels).unwrap();

        Mnist {
            train_images,
            train_labels,
            test_images,
            test_labels,
        }
    }

    pub fn trains(
        &self,
    ) -> std::iter::Zip<std::slice::Chunks<u8>, std::iter::Copied<std::slice::Iter<u8>>> {
        self.train_images
            .chunks(28 * 28)
            .zip(self.train_labels.iter().copied())
    }

    pub fn tests(
        &self,
    ) -> std::iter::Zip<std::slice::Chunks<u8>, std::iter::Copied<std::slice::Iter<u8>>> {
        self.test_images
            .chunks(28 * 28)
            .zip(self.test_labels.iter().copied())
    }
}

fn main() {}
