use crate::cnn::activation::Activation;
use ndarray::{Array, Array1, Dimension, Order};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ActivationLayer {
    #[serde(skip)]
    input: Array1<f64>,
    input_shape: Vec<usize>,
    activation: Activation,
}

impl ActivationLayer {
    pub fn new(activation: Activation, input_shape: Vec<usize>) -> Self {
        ActivationLayer {
            activation,
            input_shape,
            input: Array1::zeros(0),
        }
    }

    pub fn forward_propagate<D>(
        &mut self,
        input: &Array<f64, D>,
        _is_training: bool,
    ) -> Array<f64, D>
    where
        D: Dimension,
    {
        self.input = input.flatten_with_order(Order::RowMajor).to_owned();
        let mut output = input.to_owned();
        output
            .par_iter_mut()
            .for_each(|inp| *inp = self.activation.activate(*inp));
        output
    }

    pub fn backward_propagate<D>(&self, mut error: Array<f64, D>, _lr: f64) -> Array<f64, D>
    where
        D: Dimension,
    {
        error
            .iter_mut()
            .zip(&self.input)
            .par_bridge()
            .for_each(|(err, inp)| *err *= self.activation.deactivate(*inp));
        error
    }

    pub fn input_shape(&self) -> &Vec<usize> {
        &self.input_shape
    }
}
