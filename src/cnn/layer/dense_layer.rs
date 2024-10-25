use crate::config::DENSE_WEIGHT_SCALE;
use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Uniform;
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input_size: usize,
    output_size: usize,
    #[serde(skip)]
    input: Array2<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, bias: f64) -> Self {
        let limit = (3.0 * (DENSE_WEIGHT_SCALE / input_size as f64)).sqrt();
        let normal_distr = Uniform::new(-limit, limit);
        let mut rng = rand::thread_rng();

        let mut weights: Array2<f64> = Array2::zeros((output_size, input_size));
        weights.iter_mut().for_each(|val| {
            *val = rng.sample(normal_distr);
        });
        let biases: Array1<f64> = Array1::from_elem(output_size, bias);

        DenseLayer {
            weights,
            biases,
            output_size,
            input: Array2::zeros((0, 0)),
            input_size,
        }
    }

    pub fn forward_propagate(&mut self, input: Array2<f64>, _is_training: bool) -> Array2<f64> {
        self.input = input;
        self.calculate_output(&self.input)
    }

    pub fn backward_propagate(&mut self, error: Array2<f64>, lr: f64) -> Array2<f64> {
        let next_error = self.calculate_next_err(&error);

        self.weights -= &(self.calculate_delta_w(&error) * lr);
        self.biases -= &(error.mean_axis(Axis(0)).unwrap() * lr);
        next_error
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    fn calculate_output(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros((input.shape()[0], self.output_size));
        output
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(op_i, mut op)| {
                op.indexed_iter_mut().par_bridge().for_each(|(i, val)| {
                    let mut wi = 0.0;
                    for j in 0..self.input_size {
                        wi += self.weights[[i, j]] * input[[op_i, j]];
                    }
                    *val = wi + self.biases[[i]];
                });
            });
        output
    }

    fn calculate_next_err(&self, err: &Array2<f64>) -> Array2<f64> {
        let mut next_error: Array2<f64> = Array2::zeros(self.input.raw_dim());
        next_error
            .outer_iter_mut()
            .zip(err.outer_iter())
            .par_bridge()
            .for_each(|(mut n_err, err)| {
                n_err
                    .indexed_iter_mut()
                    .par_bridge()
                    .for_each(|(x, n_err)| {
                        *n_err = self.weights.slice(s![.., x]).dot(&err);
                    });
            });
        next_error
    }

    fn calculate_delta_w(&self, err: &Array2<f64>) -> Array2<f64> {
        let mut weight_grads: Array3<f64> =
            Array3::zeros((self.input.shape()[0], self.output_size, self.input_size));

        weight_grads
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(wei_i, mut wei_grd)| {
                wei_grd
                    .outer_iter_mut()
                    .zip(0..self.output_size)
                    .par_bridge()
                    .for_each(|(mut row, row_i)| {
                        row.indexed_iter_mut().for_each(|(col_i, col)| {
                            *col = err[[wei_i, row_i]] * self.input[[wei_i, col_i]];
                        })
                    });
            });

        weight_grads.mean_axis(Axis(0)).unwrap()
    }
}
