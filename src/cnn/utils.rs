use ndarray::{Array, Dimension, NdIndex};

pub fn batch_mean<D>(input: &[Array<f64, D>]) -> Array<f64, D>
where
    D: Dimension,
    D::Pattern: NdIndex<D>,
{
    let mut mean_arr: Array<f64, D> = Array::zeros(input[0].raw_dim());
    mean_arr.indexed_iter_mut().for_each(|(idx, m)| {
        input.iter().for_each(|inp| *m += inp[idx.clone()]);
        *m /= input.len() as f64;
    });
    mean_arr
}
