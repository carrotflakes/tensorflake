use std::sync::{Arc, Weak};

use crate::{graph::collect_function_calls, ComputedNDA};

pub fn export_dot(vars: &[ComputedNDA], file: &str) -> Result<(), std::io::Error> {
    let f = std::fs::File::create(file).unwrap();
    let mut w = std::io::BufWriter::new(f);

    write_dot(&mut w, vars, &mut default_var_printer)
}

pub fn write_dot(
    w: &mut impl std::io::Write,
    vars: &[ComputedNDA],
    var_printer: &mut impl FnMut(&ComputedNDA) -> String,
) -> Result<(), std::io::Error> {
    let fcs = collect_function_calls(vars.to_vec());
    let mut vars = fcs
        .iter()
        .flat_map(|fc| fc.xs.iter())
        .chain(vars.iter())
        .cloned()
        .collect::<Vec<_>>();
    vars.dedup();

    writeln!(w, "digraph g {{")?;

    for v in vars {
        let v_id = Arc::as_ptr(&v.inner) as usize;
        writeln!(
            w,
            "{} [label={:?} color=orange, style=filled]",
            v_id,
            var_printer(&v)
        )?;
    }

    for fc in fcs.iter() {
        let fc_id = Arc::as_ptr(fc) as usize;
        let fc_name = fc.backward.get_function_name();
        writeln!(
            w,
            "{} [label={:?} color=lightblue, style=filled, shape=box]",
            fc_id, fc_name
        )?;

        for v in fc.xs.iter() {
            let v_id = Arc::as_ptr(&v.inner) as usize;
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

pub fn default_var_printer(var: &ComputedNDA) -> String {
    var.get_name()
}

#[test]
fn test() {
    use crate::{backprop, scalar};

    let a = backprop(scalar(2.0)).named("a");
    let b = backprop(scalar(3.0)).named("b");
    let y = (a * b).named("y");

    // export_dot(&y, "graph.dot").unwrap();

    let mut w = Vec::new();
    write_dot(&mut w, &[y.clone()], &mut default_var_printer).unwrap();
    println!("{}", String::from_utf8(w).unwrap());

    // print variable values
    let mut w = Vec::new();
    write_dot(&mut w, &[y.clone()], &mut |v| {
        format!("{} {}", v.get_name(), (*v).to_string())
    })
    .unwrap();
    println!("{}", String::from_utf8(w).unwrap());
}
