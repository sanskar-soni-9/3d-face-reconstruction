use crate::{cnn::utils::batch_mean, config::DENSE_WEIGHT_SCALE};
use ndarray::{s, Array1, Array2};
use rand::Rng;
use rand_distr::Uniform;
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    output_size: usize,
    dropout_rate: f64,
    #[serde(skip)]
    input: Vec<Array1<f64>>,
    input_size: usize,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, dropout_rate: f64, bias: f64) -> Self {
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
            dropout_rate,
            input: vec![],
            input_size,
        }
    }

    pub fn forward_propagate(
        &mut self,
        input: Vec<Array1<f64>>,
        _is_training: bool,
    ) -> Vec<Array1<f64>> {
        self.input = input;
        let mut output: Vec<Array1<f64>> = vec![];
        self.input
            .iter()
            .for_each(|_| output.push(Array1::zeros(self.output_size)));

        output
            .par_iter_mut()
            .zip(&self.input)
            .for_each(|(output_arr, input_arr)| self.calculate_output(input_arr, output_arr));

        output
    }

    pub fn backward_propagate(&mut self, error: Vec<Array1<f64>>, lr: f64) -> Vec<Array1<f64>> {
        let mut next_error: Vec<Array1<f64>> = vec![];
        let mut weight_grads: Vec<Array2<f64>> = vec![];
        error.iter().for_each(|_| {
            next_error.push(Array1::zeros(self.input_size));
            weight_grads.push(Array2::zeros(self.weights.raw_dim()));
        });

        next_error
            .par_iter_mut()
            .zip(&error)
            .for_each(|(next_err, err)| self.calculate_next_err(err, next_err));

        weight_grads
            .par_iter_mut()
            .zip(0..error.len())
            .for_each(|(wg, wg_i)| self.calculate_delta_w(&error[wg_i], &self.input[wg_i], wg));

        self.weights -= &(batch_mean(&weight_grads) * lr);

        self.biases -= &(batch_mean(&error) * lr);

        next_error
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }

    fn calculate_output(&self, input: &Array1<f64>, output: &mut Array1<f64>) {
        output.indexed_iter_mut().par_bridge().for_each(|(i, val)| {
            let mut wi = 0.0;
            for j in 0..self.input_size {
                wi += self.weights[[i, j]] * input[[j]];
            }
            *val = wi + self.biases[[i]];
        });
    }

    fn calculate_next_err(&self, err: &Array1<f64>, next_err: &mut Array1<f64>) {
        next_err
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|(x, n_err)| {
                *n_err = self.weights.slice(s![.., x]).dot(err);
            });
    }

    fn calculate_delta_w(&self, err: &Array1<f64>, inp: &Array1<f64>, delta_w: &mut Array2<f64>) {
        delta_w
            .outer_iter_mut()
            .zip(0..self.output_size)
            .par_bridge()
            .for_each(|(mut row, row_i)| {
                row.indexed_iter_mut().for_each(|(col_i, col)| {
                    *col = err[[row_i]] * inp[[col_i]];
                })
            });
    }
}
