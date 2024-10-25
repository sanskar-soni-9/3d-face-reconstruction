use ndarray::{s, Array2, Array4};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct GlobalAvgPoolingLayer {
    input_shape: (usize, usize, usize, usize),
}

impl GlobalAvgPoolingLayer {
    pub fn new(input_shape: (usize, usize, usize, usize)) -> Self {
        GlobalAvgPoolingLayer { input_shape }
    }

    pub fn forward_propagate(&mut self, input: &Array4<f64>, _is_training: bool) -> Array2<f64> {
        self.calculate_output(input)
    }

    pub fn backward_propagate(&self, error: &Array2<f64>) -> Array4<f64> {
        self.calculate_next_err(error)
    }

    pub fn output_shape(&self) -> (usize, usize) {
        (self.input_shape.0, self.input_shape.1)
    }

    fn calculate_output(&self, input: &Array4<f64>) -> Array2<f64> {
        let mut output: Array2<f64> = Array2::zeros((self.input_shape.0, self.input_shape.1));
        output
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(idx, mut op)| {
                op.indexed_iter_mut().par_bridge().for_each(|(i, out)| {
                    *out = input
                        .slice(s![idx, i, .., ..])
                        .mean()
                        .expect("Average array is empty?")
                });
            });
        output
    }

    fn calculate_next_err(&self, err: &Array2<f64>) -> Array4<f64> {
        let avg_prime = 1.0 / (self.input_shape.2 * self.input_shape.3) as f64;
        let mut next_error: Array4<f64> = Array4::zeros(self.input_shape);
        next_error
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(idx, mut n_err)| {
                n_err
                    .outer_iter_mut()
                    .zip(0..self.input_shape.0)
                    .par_bridge()
                    .for_each(|(mut e, i)| e += err[[idx, i]] * avg_prime);
            });

        next_error
    }
}
