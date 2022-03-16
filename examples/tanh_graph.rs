/*
dot tanh_1.dot -T png -o tanh_1.png
dot tanh_2.dot -T png -o tanh_2.png
dot tanh_3.dot -T png -o tanh_3.png
 */

use ruzero::{functions::*, *};

fn main() {
    let x = Variable::<ENABLE_BACKPROP>::new(scalar(1.0)).named("x");
    let y = call!(Tanh, x).named("y");

    y.backward(false, true);

    for i in 1..=3 {
        let gx = x
            .get_grad::<ENABLE_BACKPROP>()
            .unwrap()
            .clone()
            .named(format!("gx{}", i));
        x.clear_grad();
        gx.backward(false, true);

        export_dot::export_dot(&[gx], &format!("tanh_{}.dot", i)).unwrap();
    }
}
