use ndarray::{Array, Dimension};
use rayon::iter::{ParallelBridge, ParallelIterator};

pub fn vec_f32_to_array_f64<const N: usize>(vec: &[f32]) -> [f64; N] {
    let mut vec_iter = vec.iter();
    [0.0; N].map(|_| {
        *vec_iter
            .next()
            .unwrap_or_else(|| panic!("Expected a Vec of length {} but it was {}", N, vec.len()))
            as f64
    })
}

pub fn vec_i16_to_array_f64<const N: usize>(vec: &[i16]) -> [f64; N] {
    let mut vec_iter = vec.iter();
    [0.0; N].map(|_| {
        *vec_iter
            .next()
            .unwrap_or_else(|| panic!("Expected a Vec of length {} but it was {}", N, vec.len()))
            as f64
    })
}

pub fn vec_to_array_f64<const N: usize>(vec: &Vec<f64>) -> [f64; N] {
    vec.to_owned().try_into().unwrap_or_else(|v: Vec<f64>| {
        panic!("Expected a Vec of length {} but it was {}", N, v.len())
    })
}

pub fn str_to_vec_f64(s: &str) -> Result<Vec<f64>, std::num::ParseFloatError> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect()
}

pub fn elementwise_max<D>(mtrx_a: &Array<f64, D>, mtrx_b: &Array<f64, D>) -> Array<f64, D>
where
    D: Dimension,
{
    let mut max = mtrx_a.to_owned();
    max.iter_mut()
        .zip(mtrx_b.iter())
        .par_bridge()
        .for_each(|(val_max, val_b)| *val_max = val_max.max(*val_b));
    max
}
