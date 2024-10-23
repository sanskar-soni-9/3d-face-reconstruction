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

    pub fn forward_propagate(
        &mut self,
        input: &Vec<Array3<f64>>,
        _is_training: bool,
    ) -> Vec<Array1<f64>> {
        let mut output: Vec<Array1<f64>> = vec![];
        input
            .iter()
            .for_each(|_| output.push(Array1::zeros(self.input_shape.0)));

        output
            .par_iter_mut()
            .zip(input)
            .for_each(|(output_arr, input_arr)| self.calculate_output(input_arr, output_arr));

        output
    }

    pub fn backward_propagate(&self, error: &Vec<Array1<f64>>) -> Vec<Array3<f64>> {
        let avg_prime = 1.0 / (self.input_shape.1 * self.input_shape.2) as f64;
        let mut next_error: Vec<Array3<f64>> = vec![];
        error
            .iter()
            .for_each(|_| next_error.push(Array3::zeros(self.input_shape)));

        next_error
            .par_iter_mut()
            .zip(error)
            .for_each(|(next_err, err)| self.calculate_next_err(err, avg_prime, next_err));

        next_error
    }

    pub fn output_size(&self) -> usize {
        self.input_shape.0
    }

    fn calculate_output(&self, input: &Array3<f64>, output: &mut Array1<f64>) {
        output.indexed_iter_mut().par_bridge().for_each(|(i, out)| {
            *out = input
                .slice(s![i, .., ..])
                .mean()
                .expect("Average array is empty?")
        });
    }

    fn calculate_next_err(&self, err: &Array1<f64>, avg_prime: f64, next_err: &mut Array3<f64>) {
        next_err
            .outer_iter_mut()
            .zip(0..self.input_shape.0)
            .par_bridge()
            .for_each(|(mut e, i)| e += err[[i]] * avg_prime);
    }
}
