use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{call, functions::Add, Funcall, Function, Tensor, Variable};

pub enum RecorderInner {
    Owned { funcalls: Vec<Funcall> },
    Ref(Recorder),
    None,
}

pub struct Recorder {
    pub(crate) inner: Arc<Mutex<RecorderInner>>,
}

impl Recorder {
    pub fn new() -> Self {
        Recorder {
            inner: Arc::new(Mutex::new(RecorderInner::Owned {
                funcalls: Vec::new(),
            })),
        }
    }

    pub fn get_owned(&self) -> Recorder {
        let inner = self.inner.lock().unwrap();
        match *inner {
            RecorderInner::Owned { .. } => self.clone(),
            RecorderInner::Ref(ref recorder) => recorder.get_owned(),
            RecorderInner::None => panic!("do not get_owned()"),
        }
    }

    pub fn is_prevented(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        match *inner {
            RecorderInner::Owned { .. } => false,
            RecorderInner::Ref(ref recorder) => recorder.is_prevented(),
            RecorderInner::None => true,
        }
    }

    pub fn merge(mut recorders: Vec<Recorder>) -> Self {
        recorders = recorders.into_iter().map(|r| r.get_owned()).collect();
        recorders.dedup();
        if recorders.len() == 1 {
            return recorders.pop().unwrap();
        }
        let recorder = Recorder::new();

        let fcs: Vec<_> = recorders
            .into_iter()
            .flat_map(|r| {
                let mut inner = r.inner.lock().unwrap();
                let fcs = match *inner {
                    RecorderInner::Owned { ref mut funcalls } => {
                        funcalls.drain(..).collect::<Vec<_>>()
                    }
                    _ => unreachable!(),
                };
                *inner = RecorderInner::Ref(recorder.clone());
                fcs
            })
            .collect();

        match *recorder.inner.lock().unwrap() {
            RecorderInner::Owned { ref mut funcalls } => {
                funcalls.extend(fcs);
            }
            _ => unreachable!(),
        }

        recorder
    }

    pub fn push(&self, funcall: Funcall) {
        match *self.get_owned().inner.lock().unwrap() {
            RecorderInner::Owned { ref mut funcalls } => funcalls.push(funcall),
            RecorderInner::Ref(ref r) => r.push(funcall),
            RecorderInner::None => panic!("do not push()"),
        }
    }

    pub fn gradients(self, ys: &Vec<Variable>, xs: &Vec<Variable>) -> Vec<Variable> {
        let owned = self.get_owned();

        let mut grads = HashMap::new();

        for y in ys.iter() {
            grads.insert(
                Arc::as_ptr(&y.inner),
                Variable::new(Tensor::ones(y.shape())),
            );
        }

        // let mut fcs = Vec::new();
        // match *owned.inner.lock().unwrap() {
        //     RecorderInner::Owned { ref mut funcalls } => fcs = std::mem::replace(funcalls, fcs),
        //     _ => todo!(),
        // }
        let fcs = if let RecorderInner::Owned { funcalls } =std::mem::replace(&mut *owned.inner.lock().unwrap(), RecorderInner::None) {
            funcalls
        } else {
            unreachable!()
        };

        for fc in fcs.iter().rev() {
            let gys = fc
                .ys
                .iter()
                .map(|y| grads.get(&Arc::as_ptr(&y.inner)).cloned())
                .collect::<Option<Vec<_>>>();
            if let Some(gys) = gys {
                let gxs = fc.backward.backward(&fc.xs, &fc.ys, &gys);
                for (x, gx) in fc.xs.iter().zip(gxs.iter()) {
                    match grads.entry(Arc::as_ptr(&x.inner)) {
                        std::collections::hash_map::Entry::Occupied(mut entry) => {
                            *entry.get_mut() = call!(Add, entry.get(), gx);
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(gx.clone());
                        }
                    }
                }
                // TODO: waste grad
            }
        }

        // match *owned.inner.lock().unwrap() {
        //     RecorderInner::Owned { ref mut funcalls } => {
        //         funcalls.splice(0..0, fcs);
        //     }
        //     _ => unreachable!(),
        // }
        std::mem::replace(&mut *owned.inner.lock().unwrap(), RecorderInner::Owned { funcalls: fcs }); // TODO

        xs.iter()
            .map(|x| grads.get(&Arc::as_ptr(&x.inner)).unwrap().clone())
            .collect()
    }
}

impl Clone for Recorder {
    fn clone(&self) -> Self {
        Recorder {
            inner: self.inner.clone(),
        }
    }
}

impl PartialEq for Recorder {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

pub fn gradients(ys: &Vec<Variable>, xs: &Vec<Variable>) -> Vec<Variable> {
    let recorder = xs[0].inner.attrs.lock().unwrap().recorder.clone().unwrap();
    recorder.gradients(ys, xs)
}

// pub fn clear_grads(x: &Variable) {
//     let recorder = x.inner.attrs.lock().unwrap().recorder.clone().unwrap();
//     let owned = recorder.get_owned();
//     let mut owned = owned.inner.lock().unwrap();
//     match *owned {
//         RecorderInner::Owned { ref mut funcalls } => {
//             funcalls.clear();
//         }
//         RecorderInner::Ref(_) => unreachable!(),
//     }
// }
