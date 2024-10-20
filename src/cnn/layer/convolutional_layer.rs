use super::{relu, relu_prime};
use ndarray::{s, Array3, Array4, Axis};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ops::AddAssign;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConvolutionalLayer {
    kernel_shape: (usize, usize, usize, usize),
    strides: usize,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize), // (filters, height, width)
    kernels: Array4<f64>,
    #[serde(skip)]
    output: Array3<f64>,
    #[serde(skip)]
    input: Array3<f64>,
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
}

impl ConvolutionalLayer {
    pub fn new(
        filters: usize,
        kernel_size: usize,
        strides: usize,
        mut input_shape: (usize, usize, usize),
        padding: bool,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (kernel_size * kernel_size * input_shape.0) as f64).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();
        let kernels =
            Array4::from_shape_fn((filters, input_shape.0, kernel_size, kernel_size), |_| {
                rng.sample(normal_distr)
            });

        let mut output_shape = (
            filters,
            (input_shape.1 - kernel_size) / strides + 1,
            (input_shape.2 - kernel_size) / strides + 1,
        );

        if padding {
            output_shape = (filters, input_shape.1 / strides, input_shape.2 / strides);
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

        ConvolutionalLayer {
            kernel_shape: (filters, input_shape.0, kernel_size, kernel_size),
            strides,
            input_shape,
            kernels,
            output_shape,
            output: Array3::zeros((0, 0, 0)),
            input: Array3::zeros((0, 0, 0)),
            padding,
        }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array3<f64> {
        self.input = Array3::zeros(self.input_shape);
        self.input
            .slice_mut(s![
                ..,
                self.padding.2..self.input_shape.1 - self.padding.3,
                self.padding.0..self.input_shape.2 - self.padding.1,
            ])
            .assign(input);

        self.output = Array3::zeros(self.output_shape);
        let mut indexed_output_iter: Vec<((usize, usize, usize), &mut f64)> =
            self.output.indexed_iter_mut().collect();
        indexed_output_iter
            .par_iter_mut()
            .for_each(|((f, mut y, mut x), output_val)| {
                y *= self.strides;
                x *= self.strides;
                **output_val = relu(
                    (&self.input.slice(s![
                        ..,
                        y..y + self.kernel_shape.2,
                        x..x + self.kernel_shape.3
                    ]) * &self.kernels.slice(s![*f, .., .., ..]))
                        .sum(),
                );
            });
        self.output.clone()
    }

    pub fn backward_propagate(&mut self, mut error: Array3<f64>, lr: f64) -> Array3<f64> {
        error
            .iter_mut()
            .zip(self.output.iter())
            .par_bridge()
            .for_each(|(e, o)| *e *= relu_prime(*o));

        let mut next_error: Array3<f64> = Array3::zeros(self.input_shape);
        next_error
            .axis_iter_mut(Axis(1))
            .zip(0..self.input_shape.1)
            .par_bridge()
            .for_each(|(mut row, row_i)| {
                if row_i < self.padding.2 {
                    return;
                }
                for kernel_row in 0..self.kernel_shape.2 {
                    if kernel_row > row_i
                        || self.kernel_shape.2 - kernel_row + row_i > self.input_shape.1
                        || (row_i - kernel_row) as f64 % self.strides as f64 != 0.0
                    {
                        continue;
                    }

                    for col_i in
                        (0..self.input_shape.2 - self.kernel_shape.3 + 1).step_by(self.strides)
                    {
                        for filter in 0..self.kernel_shape.0 {
                            row.slice_mut(s![.., col_i..col_i + self.kernel_shape.3])
                                .add_assign(
                                    &(&self.kernels.slice(s![filter, .., kernel_row, ..])
                                        * error[[
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
            unpadded_next_error.assign(&next_error.slice(s![
                ..,
                self.padding.2..self.input_shape.1 - self.padding.3,
                self.padding.0..self.input_shape.2 - self.padding.1,
            ]));
            next_error = unpadded_next_error;
        }

        let mut delta_k = Array4::zeros(self.kernel_shape);
        delta_k
            .outer_iter_mut()
            .into_par_iter()
            .zip(0..self.kernel_shape.0)
            .for_each(|(mut kernel, kernel_i)| {
                for row in (0..self.input_shape.1 - self.kernel_shape.2 + 1).step_by(self.strides) {
                    for col in
                        (0..self.input_shape.2 - self.kernel_shape.3 + 1).step_by(self.strides)
                    {
                        kernel.add_assign(
                            &(&self.input.slice(s![
                                ..,
                                row..row + self.kernel_shape.2,
                                col..col + self.kernel_shape.3
                            ]) * error[[kernel_i, row / self.strides, col / self.strides]]),
                        );
                    }
                }
            });
        self.kernels -= &(&delta_k * lr);

        next_error
    }

    pub fn output_shape(&self) -> (usize, usize, usize) {
        self.output_shape
    }
}
