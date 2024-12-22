use ndarray::{Array2, Array4, Order};
use rayon::iter::{ParallelBridge, ParallelIterator};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FlattenLayer {
    input_shape: (usize, usize, usize, usize),
}

impl FlattenLayer {
    pub fn new(input_shape: (usize, usize, usize, usize)) -> Self {
        FlattenLayer { input_shape }
    }

    pub fn forward_propagate(&mut self, input: &Array4<f64>, _is_training: bool) -> Array2<f64> {
        let (batch_size, d, h, w) = self.input_shape;
        let mut output: Array2<f64> = Array2::zeros((batch_size, d * h * w));
        output
            .outer_iter_mut()
            .zip(input.outer_iter())
            .par_bridge()
            .for_each(|(mut op, inp)| {
                op.assign(&inp.flatten_with_order(Order::RowMajor).to_owned())
            });
        output
    }

    pub fn backward_propagate(&self, error: &Array2<f64>) -> Array4<f64> {
        let mut next_err: Array4<f64> = Array4::zeros(self.input_shape);
        next_err
            .outer_iter_mut()
            .zip(error.outer_iter())
            .par_bridge()
            .for_each(|(mut n_err, err)| {
                n_err.assign(
                    &err.into_shape_with_order((
                        self.input_shape.1,
                        self.input_shape.2,
                        self.input_shape.3,
                    ))
                    .unwrap(),
                )
            });
        next_err
    }

    pub fn output_shape(&self) -> (usize, usize) {
        (
            self.input_shape.0,
            self.input_shape.1 * self.input_shape.2 * self.input_shape.3,
        )
    }
}
