use crate::config::DENSE_WEIGHT_SCALE;
use ndarray::{s, Array1, Array2, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
    #[serde(skip)]
    input: Array2<f64>,
}

impl DenseLayer {
    pub fn new(input_shape: (usize, usize), output_size: usize, bias: f64) -> Self {
        let limit = (DENSE_WEIGHT_SCALE / input_shape.1 as f64).sqrt();
        let normal_distr = Normal::new(0., limit).unwrap();
        let mut rng = rand::thread_rng();

        let mut weights: Array2<f64> = Array2::zeros((output_size, input_shape.1));
        weights.iter_mut().for_each(|val| {
            *val = rng.sample(normal_distr);
        });
        let biases: Array1<f64> = Array1::from_elem(output_size, bias);

        DenseLayer {
            weights,
            biases,
            output_shape: (input_shape.0, output_size),
            input: Array2::zeros((0, 0)),
            input_shape,
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

    pub fn output_shape(&self) -> (usize, usize) {
        self.output_shape
    }

    fn calculate_output(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut output = Array2::zeros(self.output_shape);
        output
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(op_i, mut op)| {
                op.indexed_iter_mut().par_bridge().for_each(|(val_i, val)| {
                    *val = self
                        .weights
                        .slice(s![val_i, ..])
                        .dot(&input.slice(s![op_i, ..]))
                        + self.biases[[val_i]];
                });
            });
        output
    }

    fn calculate_next_err(&self, err: &Array2<f64>) -> Array2<f64> {
        let mut next_error = Array2::zeros(self.input.raw_dim());
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
        let mut weight_grads: Array3<f64> = Array3::zeros((
            self.output_shape.0,
            self.weights.shape()[0],
            self.weights.shape()[1],
        ));
        weight_grads
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(wei_i, mut wei_grd)| {
                wei_grd
                    .outer_iter_mut()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(row_i, mut row)| {
                        let row_err = err[[wei_i, row_i]];
                        row.indexed_iter_mut()
                            .for_each(|(col_i, col)| *col = self.input[[wei_i, col_i]] * row_err)
                    });
            });
        weight_grads.mean_axis(Axis(0)).unwrap()
    }
}
