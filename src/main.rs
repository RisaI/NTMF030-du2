use nalgebra::*;
use num_complex::Complex64 as C64;
use plotly::{common::Title, Layout, Plot, Scatter};
use rayon::prelude::*;
use std::f64::consts::PI;

const REDUCED_MASS: f64 = 1.;
const HBAR: f64 = 1.;

/// The chaos threshold constant
const F: f64 = 34567. / PI.powi(8);

fn v(i: usize, j: usize, r: f64) -> f64 {
    match (i, j) {
        (1, 1) => -3. * (-r.powi(2) / 4.).exp() + (-(r - 3.).powi(2)).exp(),
        (2, 2) => -3. * (-r.powi(2) / 4.).exp() + 3.5,
        (1, 2) | (2, 1) => 0.5 * (-r.powi(2) / 2.).exp(),
        _ => panic!(),
    }
}

/// Task 1
fn plot_potential() {
    let mut plot = Plot::new();
    plot.set_layout(Layout::new().title(Title::new("$\\text{Potential } V_{ij}$")));

    fn create_trace<F: Fn(f64) -> f64>(name: &str, func: F) -> Box<Scatter<f64, f64>> {
        let range = (0..60).map(|i| i as f64 * 0.1);
        Scatter::new(
            range.clone().collect::<Vec<_>>(),
            range.map(func).collect::<Vec<_>>(),
        )
        .name(name)
    }

    plot.add_traces(vec![
        create_trace("$ V_{11} $", |r| v(1, 1, r)),
        create_trace("$ V_{12} $", |r| v(1, 2, r)),
        create_trace("$ V_{22} $", |r| v(2, 2, r)),
    ]);

    plot.show();
}

/// V(x)
fn v_mat(r: f64) -> Matrix2<f64> {
    matrix! [
        v(1, 1, r), v(1, 2, r);
        v(2, 1, r), v(2, 2, r);
    ]
}

/// V_debug(x)
#[allow(dead_code)]
fn v_debug_mat(_r: f64) -> Matrix2<f64> {
    matrix! [
        0., 0.;
        0., 1.;
    ]
}

/// Q(x)
fn q_mat(e: f64, r: f64) -> Matrix2<f64> {
    let prefactor = 2. * REDUCED_MASS / HBAR.powi(2);

    // Q(x) = (2 mu / hbar^2) * [ E I - V(x) ]
    prefactor * (e * Matrix2::identity() - v_mat(r))
}

/// T(x)
fn t_mat(e: f64, r: f64, h: f64) -> Matrix2<f64> {
    let prefactor = -(h.powi(2) / 12.);

    prefactor * q_mat(e, r)
}

/// F(x)
fn u_mat(e: f64, r: f64, h: f64) -> Matrix2<C64> {
    let t_n = t_mat(e, r, h).map(C64::from);

    (Matrix2::identity() * C64::from(2.) + t_n * C64::from(10.))
        * (Matrix2::identity() - t_n).try_inverse().unwrap()
}

fn get_r_matrix(e: f64, r0: f64, n: usize) -> Matrix2<C64> {
    assert!(n > 0);

    let h = r0 / n as f64;

    // R_{n-1}^{-1}
    let mut inv_r_prev = Matrix2::zeros();
    let mut r_n = Matrix2::zeros();

    for i in 0..=n {
        if i > 0 {
            inv_r_prev = r_n.try_inverse().unwrap();
        }

        r_n = u_mat(e, h * i as f64, h) - inv_r_prev;
    }

    // 2h (R_n - R_{n-1}^{-1})^{-1}
    let t = Matrix2::identity() - t_mat(e, r0, h).map(C64::from);

    t.try_inverse().unwrap() * (r_n - inv_r_prev).try_inverse().unwrap() * C64::from(2. * h) * t
}

#[allow(dead_code)]
fn get_analytic_r_matrix(e: f64, r0: f64) -> Matrix2<C64> {
    let [v_11, v_22] = [0, 1].map(|i| v_debug_mat(r0)[(i, i)]).map(|v_ii| {
        if v_ii < e {
            let k_i = (2. * (e - v_ii)).sqrt();

            (k_i * r0).tan() / k_i
        } else {
            let k_i = (2. * (v_ii - e)).sqrt();

            (k_i * r0).tanh() / k_i
        }
    });

    matrix![
        v_11, 0.;
        0., v_22;
    ]
    .map(C64::from)
}

/// \hat{j}_0
fn f(k: C64, z: C64) -> C64 {
    if k.re.abs() < k.im.abs() {
        (k * z).sin() / C64::i()
    } else {
        (k * z).sin()
    }
}
fn df(k: C64, z: C64) -> C64 {
    k * (k * z).cos()
}

/// \hat{n}_0
fn g(k: C64, z: C64) -> C64 {
    -(k * z).cos()
}

fn dg(k: C64, z: C64) -> C64 {
    k * if k.re.abs() < k.im.abs() {
        (k * z).sin() * C64::i()
    } else {
        (k * z).sin()
    }
}

fn func_mat<F: Fn(C64, C64) -> C64>(e: f64, r0: f64, f: F) -> Matrix2<C64> {
    const LIMIT: f64 = 1e2;

    let [p_1, p_2] = [1, 2].map(|i| C64::from(2. * (e - v(i, i, LIMIT))).sqrt());

    matrix![
        f(p_1, r0.into()), C64::from(0.);
        C64::from(0.), f(p_2, r0.into());
    ]
}

fn get_phases_and_cross_sections(e: f64, r0: f64, n: usize) -> [f64; 4] {
    let r = get_r_matrix(e, r0, n);

    // (f - df * r) * (g - dg * r)^{-1}
    let k = -(func_mat(e, r0, f) - func_mat(e, r0, df) * r)
        * (func_mat(e, r0, g) - func_mat(e, r0, dg) * r)
            .try_inverse()
            .unwrap();
    let k = k.map(|v| C64::from(v.re));

    // Explicit form of S-matrix through K-matrix
    let s = (Matrix2::identity() + k * C64::i())
        * (Matrix2::identity() - k * C64::i()).try_inverse().unwrap();

    let deltas = s
        .eigenvalues()
        .unwrap()
        .map(|ev| (ev.ln() / (2. * C64::i())).re);

    // This is one of the nastiest and most bizzare hacks
    // I've ever had to write during my 11 years of programming.
    // The eigenvalues of both the K-matrix and the S-matrix
    // tend to get entangled and their connection to a specific
    // scatter channel gets lost by this effect.
    // Through many hours of peering through time and space
    // I've come up with a very specific way of determining these
    // eigenspace crossing points and this will correctly order
    // the eigenvalues to correlate with their respective channel.
    // I would like to thank plotly.js and it's contributors for
    // providing me the necessary tools to experimentaly detect
    // the chaotic swapping threshold F = 34567 / Pi^8 \approx 3.643
    let swap = (s[(0, 0)].re - s[(1, 1)].re) > 0. || e > F;

    [
        deltas[if swap { 1 } else { 0 }],
        deltas[if swap { 0 } else { 1 }],
        (PI / e) * (s[(0, 0)] - 1.).norm_sqr(),
        (PI / e) * (s[(0, 1)]).norm_sqr(),
    ]
}

fn main() {
    // Task 1
    // println!("Plotting potentials");
    // plot_potential();
    println!("Appears constant for r > 5.5");

    // Task 2
    // 2h psi_n = R (psi_{n+1} - psi_{n-1})
    // 2h psi_n = R (R_n - R_{n-1}^{-1}) psi_n
    // R(x_n) = 2h (R_n - R_{n-1}^{-1})^{-1}
    // Nothing to plot
    // TODO: constant potential?

    // Task 3
    println!("{:.3}", get_r_matrix(3.6, 5.5, 1000));
    // println!("{:.3}", get_analytic_r_matrix(e, 5.5));

    // Task 4 and 5
    println!("Plotting ");
    // ? sigma = (pi / k^2) sum_k | e^{2i delta_k} - 1 | = (4 pi / k^2) sum_k sin^2 delta_k

    // Run the calculation for number of POINTS equidistant in the interval [0,5] a.u.
    const POINTS: usize = 2_000;
    const FROM: f64 = 0.;

    let results: Vec<Vec<f64>> = (0..=POINTS)
        .into_par_iter()
        .map(|i| {
            let e = FROM + i as f64 * ((5. - FROM) / POINTS as f64);
            let result = get_phases_and_cross_sections(e, 5.5, 2_000);

            // Join x and y values
            std::iter::once(e).chain(result).collect()
        })
        .collect();

    // Plot the fruits of our hard work
    let get_column = move |idx: usize| results.iter().map(|row| row[idx]).collect::<Vec<_>>();
    let x_col = get_column(0);

    // Plot phase shifts
    {
        let mut plot = Plot::new();
        plot.set_layout(plotly::Layout::new().title(Title::new("Phase shifts")));

        plot.add_traces(vec![
            Scatter::new(x_col.iter().copied(), get_column(1)).name("$ \\delta_1 $"),
            Scatter::new(x_col.iter().copied(), get_column(2)).name("$ \\delta_2 $"),
        ]);

        plot.show();
    }

    // Plot cross sections
    {
        let mut plot = Plot::new();
        plot.set_layout(Layout::new().title(Title::new("Cross sections")));

        plot.add_traces(vec![
            Scatter::new(x_col.iter().copied(), get_column(3)).name("$ \\sigma_\\text{el} $"),
            Scatter::new(x_col.iter().copied(), get_column(4)).name("$ \\sigma_\\text{ex} $"),
        ]);

        plot.show();
    }
}
