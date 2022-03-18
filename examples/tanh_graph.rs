/*
dot tanh_1.dot -T png -o tanh_1.png
dot tanh_2.dot -T png -o tanh_2.png
dot tanh_3.dot -T png -o tanh_3.png
 */

use ruzero::{functions::*, *};

fn main() {
    let x = backprop(scalar(1.0)).named("x");
    let y = call!(Tanh, x).named("y");

    let mut gs = gradients(&[y.clone()], &[x.clone()], true);

    for i in 1..=3 {
        let gx = gs[0]
            .clone()
            .named(format!("gx{}", i));
            
        gs = gradients(&gs, &[x.clone()], true);

        export_dot::export_dot(&[gx], &format!("tanh_{}.dot", i)).unwrap();
    }
}
