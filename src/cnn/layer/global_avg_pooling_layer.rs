use ndarray::{s, Array1, Array3};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GlobalAvgPoolingLayer {
    input_shape: (usize, usize, usize),
}

impl GlobalAvgPoolingLayer {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        GlobalAvgPoolingLayer { input_shape }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array1<f64> {
        let mut output: Array1<f64> = Array1::<f64>::zeros(self.input_shape.0);
        output.indexed_iter_mut().par_bridge().for_each(|(i, out)| {
            *out = input
                .slice(s![i, .., ..])
                .mean()
                .expect("Average array is empty?")
        });
        output
    }

    pub fn backward_propagate(&self, error: &Array1<f64>) -> Array3<f64> {
        let avg_prime = 1.0 / (self.input_shape.1 * self.input_shape.2) as f64;
        let mut next_error: Array3<f64> = Array3::zeros(self.input_shape);
        next_error
            .outer_iter_mut()
            .zip(0..self.input_shape.0)
            .par_bridge()
            .for_each(|(mut e, i)| e += error[[i]] * avg_prime);
        next_error
    }

    pub fn output_size(&self) -> usize {
        self.input_shape.0
    }
}
