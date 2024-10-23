use crate::cnn::activation::Activation;
use ndarray::{Array, Array1, Dimension, Order};
use rayon::prelude::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ActivationLayer {
    #[serde(skip)]
    input: Vec<Array1<f64>>,
    #[serde(skip)]
    input_shape: Vec<usize>,
    activation: Activation,
}

impl ActivationLayer {
    pub fn new(activation: Activation, input_shape: Vec<usize>) -> Self {
        ActivationLayer {
            activation,
            input_shape,
            input: vec![],
        }
    }

    pub fn forward_propagate<D>(
        &mut self,
        input: &Vec<Array<f64, D>>,
        _is_training: bool,
    ) -> Vec<Array<f64, D>>
    where
        D: Dimension,
    {
        self.input.clear();
        input.iter().for_each(|inp| {
            self.input
                .push(inp.flatten_with_order(Order::RowMajor).to_owned())
        });
        let mut input = input.to_owned();
        input.par_iter_mut().for_each(|inp_arr| {
            inp_arr
                .par_iter_mut()
                .for_each(|inp| *inp = self.activation.activate(*inp))
        });
        input
    }

    pub fn backward_propagate<D>(
        &self,
        mut error: Vec<Array<f64, D>>,
        _lr: f64,
    ) -> Vec<Array<f64, D>>
    where
        D: Dimension,
    {
        error
            .iter_mut()
            .zip(&self.input)
            .par_bridge()
            .for_each(|(err_arr, inp_arr)| {
                err_arr
                    .iter_mut()
                    .zip(inp_arr)
                    .par_bridge()
                    .for_each(|(err, inp)| *err *= self.activation.deactivate(*inp))
            });
        error
    }

    pub fn input_shape(&self) -> &Vec<usize> {
        &self.input_shape
    }
}
