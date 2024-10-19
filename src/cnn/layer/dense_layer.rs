use ndarray::{s, Array1, Array2};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;

use super::{relu, relu_prime};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DenseLayer {
    weights: Array2<f64>,
    biases: Array1<f64>,
    output_size: usize,
    dropout_rate: f64,
    #[serde(skip)]
    input: Array1<f64>,
    #[serde(skip)]
    output: Array1<f64>,
    #[serde(skip)]
    dropout_mask: Array1<f64>,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, dropout_rate: f64, bias: f64) -> Self {
        let std_dev = (2.0 / input_size as f64).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();
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
            input: Array1::zeros(0),
            output: Array1::zeros(0),
            dropout_mask: Array1::zeros(0),
        }
    }

    pub fn forward_propagate(&mut self, input: &Array1<f64>, is_training: bool) -> Array1<f64> {
        self.input = input.to_owned();
        self.output = Array1::zeros(self.output_size);
        self.output
            .iter_mut()
            .zip(0..self.output_size)
            .par_bridge()
            .for_each(|(val, i)| {
                let mut wi = 0.0;
                for j in 0..self.input.len() {
                    wi += self.weights[[i, j]] * self.input[[j]];
                }
                *val = relu(wi + self.biases[[i]]);
            });

        if is_training {
            let mut rng = rand::thread_rng();
            self.dropout_mask = Array1::from_shape_fn(self.output_size, |_| {
                if rng.gen::<f64>() < self.dropout_rate {
                    0.0
                } else {
                    1.0 / (1.0 - self.dropout_rate) // Inverted Dropout
                }
            });
            self.output *= &self.dropout_mask;
        }

        self.output.clone()
    }

    pub fn backward_propagate(&mut self, mut error: Array1<f64>, lr: f64) -> Array1<f64> {
        error *= &self.dropout_mask;
        error
            .iter_mut()
            .zip(0..self.output_size)
            .par_bridge()
            .for_each(|(e, i)| *e *= relu_prime(self.output[i]));

        let mut next_error: Array1<f64> = Array1::zeros(self.input.len());
        next_error
            .iter_mut()
            .zip(0..self.input.len())
            .par_bridge()
            .for_each(|(err, x)| {
                *err = self.weights.slice(s![.., x]).dot(&error);
            });

        self.weights
            .outer_iter_mut()
            .into_par_iter()
            .zip(0..self.output_size)
            .for_each(|(mut row, row_i)| {
                row.iter_mut()
                    .zip(0..self.input.len())
                    .for_each(|(col, col_i)| {
                        *col -= error[[row_i]] * self.input[[col_i]] * lr;
                    })
            });

        self.biases
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|(i, bias)| {
                *bias -= error[[i]] * lr;
            });
        next_error
    }

    pub fn output_size(&self) -> usize {
        self.output_size
    }
}
