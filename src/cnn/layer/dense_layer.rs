use ndarray::{Array1, Array2};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;

pub struct DenseLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    pub output_size: usize,
    dropout_rate: f32,
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize, dropout_rate: f32, bias: f32) -> Self {
        let std_dev = (2.0 / input_size as f32).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();

        let mut weights: Array2<f32> = Array2::zeros((output_size, input_size));
        let Some(weights_slice) = weights.as_slice_mut() else { panic!("Unexpected error occurred!")};
        weights_slice.par_iter_mut().for_each(|val| {
            let mut rng = rand::thread_rng();
            *val = rng.sample(normal_distr);
        });
        let biases: Array1<f32> = Array1::from_elem(output_size, bias);

        DenseLayer {
            weights,
            biases,
            output_size,
            dropout_rate,
        }
    }

    pub fn forward_propagate(&mut self, input: &Array1<f32>, is_training: bool) -> Array1<f32> {
        let mut logits = self.weights.dot(input) + &self.biases;
        if is_training {
            let mut rng = rand::thread_rng();
            let dropout_mask: Array1<f32> = Array1::from_shape_fn(logits.len(), |_| {
                if rng.gen::<f32>() < self.dropout_rate {
                    0.0
                } else {
                    1.0 / (1.0 - self.dropout_rate) // Inverted Dropout
                }
            });
            logits *= &dropout_mask;
        }

        logits.map_mut(|val| *val = val.max(0.0));
        logits
    }

    pub fn backward_propagate(&mut self, error: &Vec<f32>) -> Vec<f32> {
        // TODO: implement
        error.to_owned()
    }
}
