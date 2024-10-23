use crate::{cnn::utils::batch_mean, config::CONV_WEIGHT_SCALE};
use ndarray::{s, Array3, Axis};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ops::AddAssign;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DepthwiseConvolutionalLayer {
    kernel_shape: (usize, usize, usize),
    strides: usize,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    kernels: Array3<f64>,
    #[serde(skip)]
    input: Vec<Array3<f64>>,
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
}

impl DepthwiseConvolutionalLayer {
    pub fn new(
        kernel_size: usize,
        strides: usize,
        mut input_shape: (usize, usize, usize),
        padding: bool,
    ) -> Self {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }

        let mut rng = rand::thread_rng();
        let std_dev =
            (CONV_WEIGHT_SCALE / (kernel_size * kernel_size * input_shape.0) as f64).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();
        let kernel_shape = (input_shape.0, kernel_size, kernel_size);
        let kernels = Array3::from_shape_fn(kernel_shape, |_| rng.sample(normal_distr));

        let mut output_shape = (
            input_shape.0,
            (input_shape.1 - kernel_size) / strides + 1,
            (input_shape.2 - kernel_size) / strides + 1,
        );

        if padding {
            output_shape = (
                input_shape.0,
                input_shape.1 / strides,
                input_shape.2 / strides,
            );
            input_shape = (
                input_shape.0,
                input_shape.1 + kernel_size - 1,
                input_shape.2 + kernel_size - 1,
            );
        }

        let padding = if padding {
            let padding = (kernel_size - 1) as f64 / 2.0;
            (
                padding.floor() as usize,
                padding.ceil() as usize,
                padding.floor() as usize,
                padding.ceil() as usize,
            )
        } else {
            (0, 0, 0, 0)
        };

        DepthwiseConvolutionalLayer {
            kernel_shape,
            strides,
            input_shape,
            output_shape,
            kernels,
            input: vec![],
            padding,
        }
    }

    pub fn forward_propagate(
        &mut self,
        input: &[Array3<f64>],
        _is_training: bool,
    ) -> Vec<Array3<f64>> {
        self.input.clear();
        let mut output: Vec<Array3<f64>> = vec![];

        input.iter().for_each(|inp| {
            self.input.push(self.prepare(inp));
            output.push(Array3::zeros(self.output_shape));
        });

        output
            .par_iter_mut()
            .zip(&self.input)
            .for_each(|(output_arr, input_arr)| self.calculate_output(input_arr, output_arr));

        output
    }

    pub fn backward_propagate(&mut self, error: Vec<Array3<f64>>, lr: f64) -> Vec<Array3<f64>> {
        let mut next_error: Vec<Array3<f64>> = vec![];
        let mut kernel_grads: Vec<Array3<f64>> = vec![];
        error.iter().for_each(|_| {
            next_error.push(Array3::zeros(self.input_shape));
            kernel_grads.push(Array3::zeros(self.kernel_shape));
        });

        next_error
            .par_iter_mut()
            .zip(&error)
            .for_each(|(n_err, err)| self.calculate_next_err(err, n_err));

        kernel_grads
            .par_iter_mut()
            .zip(0..error.len())
            .for_each(|(kg, kg_i)| self.calculate_delta_k(&self.input[kg_i], &error[kg_i], kg));

        self.kernels -= &(batch_mean(&kernel_grads) * lr);
        next_error
    }

    pub fn output_shape(&self) -> (usize, usize, usize) {
        self.output_shape
    }

    fn prepare(&self, input: &Array3<f64>) -> Array3<f64> {
        let mut padded_input = Array3::zeros(self.input_shape);
        padded_input
            .slice_mut(s![
                ..,
                self.padding.2..self.input_shape.1 - self.padding.3,
                self.padding.0..self.input_shape.2 - self.padding.1,
            ])
            .assign(input);
        padded_input
    }

    fn calculate_output(&self, input: &Array3<f64>, output: &mut Array3<f64>) {
        output
            .indexed_iter_mut()
            .par_bridge()
            .for_each(|((f, mut r, mut c), o)| {
                r *= self.strides;
                c *= self.strides;
                *o = (&input.slice(s![
                    f,
                    r..r + self.kernel_shape.1,
                    c..c + self.kernel_shape.2
                ]) * &self.kernels.slice(s![f, .., ..]))
                    .sum();
            });
    }

    fn calculate_next_err(&self, err: &Array3<f64>, next_err: &mut Array3<f64>) {
        next_err
            .axis_iter_mut(Axis(1))
            .zip(0..self.input_shape.1)
            .par_bridge()
            .for_each(|(mut row, row_i)| {
                if row_i < self.padding.2 {
                    return;
                }
                for kernel_row in 0..self.kernel_shape.1 {
                    if kernel_row > row_i
                        || self.kernel_shape.1 - kernel_row + row_i > self.input_shape.1
                        || (row_i - kernel_row) as f64 % self.strides as f64 != 0.0
                    {
                        continue;
                    }

                    for col_i in
                        (0..self.input_shape.2 - self.kernel_shape.2 + 1).step_by(self.strides)
                    {
                        for filter in 0..self.kernel_shape.0 {
                            row.slice_mut(s![filter, col_i..col_i + self.kernel_shape.2])
                                .add_assign(
                                    &(&self.kernels.slice(s![filter, kernel_row, ..])
                                        * err[[
                                            filter,
                                            (row_i - kernel_row) / self.strides,
                                            col_i / self.strides,
                                        ]]),
                                );
                        }
                    }
                }
            });

        if self.padding != (0, 0, 0, 0) {
            let mut unpadded_next_error: Array3<f64> = Array3::zeros((
                self.input_shape.0,
                self.input_shape.1 - self.padding.2 - self.padding.3,
                self.input_shape.2 - self.padding.0 - self.padding.1,
            ));
            unpadded_next_error.assign(&next_err.slice(s![
                ..,
                self.padding.2..self.input_shape.1 - self.padding.3,
                self.padding.0..self.input_shape.2 - self.padding.1,
            ]));
            *next_err = unpadded_next_error;
        }
    }

    fn calculate_delta_k(&self, input: &Array3<f64>, err: &Array3<f64>, delta_k: &mut Array3<f64>) {
        delta_k
            .outer_iter_mut()
            .zip(0..self.kernel_shape.0)
            .par_bridge()
            .for_each(|(mut kernel, kernel_i)| {
                for row in (0..self.input_shape.1 - self.kernel_shape.2 + 1).step_by(self.strides) {
                    for col in
                        (0..self.input_shape.2 - self.kernel_shape.2 + 1).step_by(self.strides)
                    {
                        kernel.add_assign(
                            &(&input.slice(s![
                                kernel_i,
                                row..row + self.kernel_shape.1,
                                col..col + self.kernel_shape.2
                            ]) * err[[kernel_i, row / self.strides, col / self.strides]]),
                        );
                    }
                }
            });
    }
}
