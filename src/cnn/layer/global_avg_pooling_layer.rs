use ndarray::{s, Array1, Array3};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GlobalAvgPoolingLayer {
    input_size: (usize, usize, usize),
}

impl GlobalAvgPoolingLayer {
    pub fn new(input_size: (usize, usize, usize)) -> Self {
        GlobalAvgPoolingLayer { input_size }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array1<f64> {
        let mut output: Array1<f64> = Array1::<f64>::zeros(self.input_size.0);
        output.indexed_iter_mut().par_bridge().for_each(|(i, out)| {
            *out = input
                .slice(s![i, .., ..])
                .mean()
                .expect("Average array is empty?")
        });
        output
    }

    pub fn backward_propagate(&mut self, error: &Array1<f64>) -> Array3<f64> {
        let avg_prime = 1.0 / (self.input_size.1 * self.input_size.2) as f64;
        let mut next_error: Array3<f64> = Array3::zeros(self.input_size);
        next_error
            .outer_iter_mut()
            .into_par_iter()
            .zip(0..self.input_size.0)
            .for_each(|(mut e, i)| e += error[[i]] * avg_prime);
        next_error
    }

    pub fn get_output_size(&self) -> usize {
        self.input_size.0
    }
}
