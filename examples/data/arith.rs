use ndarray_rand::rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;

enum Tree {
    Int(i32),
    Add(Box<Tree>, Box<Tree>),
    Sub(Box<Tree>, Box<Tree>),
    Mul(Box<Tree>, Box<Tree>),
}

impl Tree {
    fn random(rng: &mut Isaac64Rng, depth: usize) -> Self {
        let r = rng.gen::<u32>() % (3 + depth as u32);
        match r {
            0 => Self::Add(
                Box::new(Self::random(rng, depth + 1)),
                Box::new(Self::random(rng, depth + 1)),
            ),
            1 => Self::Sub(
                Box::new(Self::random(rng, depth + 1)),
                Box::new(Self::random(rng, depth + 1)),
            ),
            2 => Self::Mul(
                Box::new(Self::random(rng, depth + 1)),
                Box::new(Self::random(rng, depth + 1)),
            ),
            _ => Self::Int(rng.gen::<i32>().abs() % 30 - 10),
        }
    }

    fn to_string(&self, priority: usize) -> String {
        let (s, p) = match self {
            Self::Int(i) => (i.to_string(), 3),
            Self::Add(l, r) => (format!("{}+{}", l.to_string(1), r.to_string(1)), 1),
            Self::Sub(l, r) => (format!("{}-{}", l.to_string(1), r.to_string(1)), 1),
            Self::Mul(l, r) => (format!("{}*{}", l.to_string(2), r.to_string(2)), 2),
        };
        if priority >= p {
            format!("({})", s)
        } else {
            s
        }
    }

    fn eval(&self) -> Option<i32> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Add(l, r) => l.eval()?.checked_add(r.eval()?),
            Self::Sub(l, r) => l.eval()?.checked_sub(r.eval()?),
            Self::Mul(l, r) => l.eval()?.checked_mul(r.eval()?),
        }
    }
}

pub fn make(num: usize, seed: u64, depth: usize) -> Vec<String> {
    let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(seed);

    let mut v = Vec::new();
    for _ in 0..num {
        let mut t = Tree::random(&mut rng, depth);
        while t.eval().unwrap_or(1000).abs() >= 100 {
            t = Tree::random(&mut rng, 0);
        }
        v.push(format!("{}={}?", t.to_string(0), t.eval().unwrap()));
    }
    v
}

pub const CHARS: &'static str = "?0123456789()+-*=";
pub const VOCAB_SIZE: usize = CHARS.len();

#[allow(dead_code)]
pub fn encode(s: &str) -> Vec<usize> {
    let mut v = Vec::new();
    for c in s.chars() {
        v.push(CHARS.find(c).unwrap_or(0));
    }
    v
}

#[allow(dead_code)]
pub fn decode(v: &[usize]) -> String {
    let mut s = String::new();
    for c in v {
        s.push(CHARS.chars().nth(*c).expect("out of vocabulary"));
    }
    s
}

#[allow(dead_code)]
fn main() {
    println!("vocab size: {}", VOCAB_SIZE);
    for s in make(10, 42, 10) {
        println!("{}", s);
    }
}
