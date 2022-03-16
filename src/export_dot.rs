use std::rc::{Rc, Weak};

use crate::{collect_funcalls, Variable, ENABLE_BACKPROP};

pub fn export_dot(var: &Variable<ENABLE_BACKPROP>, file: &str) -> Result<(), std::io::Error> {
    let f = std::fs::File::create(file).unwrap();
    let mut w = std::io::BufWriter::new(f);

    write_dot(&mut w, var, &mut default_var_printer)
}

pub fn write_dot(
    w: &mut impl std::io::Write,
    var: &Variable<ENABLE_BACKPROP>,
    var_printer: &mut impl FnMut(&Variable<ENABLE_BACKPROP>) -> String,
) -> Result<(), std::io::Error> {
    let fcs = collect_funcalls(vec![var.clone()]);
    let mut vars = fcs
        .iter()
        .flat_map(|fc| fc.xs.iter().cloned())
        .collect::<Vec<_>>();
    vars.push(var.clone());
    vars.dedup();

    writeln!(w, "digraph g {{")?;

    for v in vars {
        let v_id = Rc::as_ptr(&v.inner) as usize;
        writeln!(
            w,
            "{} [label={:?} color=orange, style=filled]",
            v_id,
            var_printer(&v)
        )?;
    }

    for fc in fcs.iter() {
        let fc_id = Rc::as_ptr(fc) as usize;
        let fc_name = fc.function.get_function_name();
        writeln!(
            w,
            "{} [label={:?} color=lightblue, style=filled, shape=box]",
            fc_id, fc_name
        )?;

        for v in fc.xs.iter() {
            let v_id = Rc::as_ptr(&v.inner) as usize;
            writeln!(w, "{} -> {}", v_id, fc_id)?;
        }
        for v in fc.ys.iter() {
            let v_id = Weak::as_ptr(&v) as usize;
            writeln!(w, "{} -> {}", fc_id, v_id)?;
        }
    }

    writeln!(w, "}}")?;

    Ok(())
}

pub fn default_var_printer(var: &Variable<ENABLE_BACKPROP>) -> String {
    var.get_name()
}

#[test]
fn test() {
    use crate::{call, functions::Mul, scalar, Function, Variable, ENABLE_BACKPROP};

    let a = Variable::<ENABLE_BACKPROP>::new(scalar(2.0)).named("a");
    let b = Variable::<ENABLE_BACKPROP>::new(scalar(3.0)).named("b");
    let y = call!(Mul, a, b).named("y");

    // export_dot(&y, "graph.dot").unwrap();

    let mut w = Vec::new();
    write_dot(&mut w, &y, &mut default_var_printer).unwrap();
    println!("{}", String::from_utf8(w).unwrap());

    // print variable values
    let mut w = Vec::new();
    write_dot(&mut w, &y, &mut |v| {
        format!("{} {}", v.get_name(), (*v).to_string())
    })
    .unwrap();
    println!("{}", String::from_utf8(w).unwrap());
}
