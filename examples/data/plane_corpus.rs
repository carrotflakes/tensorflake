#![allow(dead_code)]

use std::{error::Error, fs::File, io::Read};

pub fn load(path: &str) -> Result<String, Box<dyn Error>> {
    let mut f = File::open(path)?;
    let mut v = String::new();
    f.read_to_string(&mut v)?;
    Ok(v)
}

pub fn windows(s: &str, size: usize, stride: usize) -> Vec<String> {
    let mut v = vec![];
    let mut i = 0;
    while i < s.len() {
        let j = (i + size).min(s.len());
        v.push(s[i..j].to_string());
        i += stride;
    }
    v
}

pub struct Vocab(Vec<char>);

impl Vocab {
    pub fn new(s: &str) -> Self {
        let mut v = Vec::new();
        for c in s.chars() {
            if v.contains(&c) {
                continue;
            }
            v.push(c);
        }
        v.sort();
        Self(v)
    }

    pub fn encode(&self, s: &str) -> Vec<usize> {
        let mut v = vec![];
        for c in s.chars() {
            v.push(self.0.iter().position(|&x| x == c).unwrap());
        }
        v
    }

    pub fn decode(&self, v: &[usize]) -> String {
        let mut s = String::new();
        for &i in v {
            s.push(self.0[i]);
        }
        s
    }

    pub fn size(&self) -> usize {
        self.0.len()
    }
}

fn main() {
    let s = load("examples/plane_corpus.rs").unwrap();
    let v = Vocab::new(&s);
    for s in windows(&s, 30, 15) {
        println!("{:?}", &s);
    }
    dbg!(v.size());
}
