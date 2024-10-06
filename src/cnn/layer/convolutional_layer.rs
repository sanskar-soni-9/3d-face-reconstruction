use crate::cnn::{relu, relu_prime};
use ndarray::{s, Array3, Array4, Axis};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ops::AddAssign;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConvolutionalLayer {
    filters: usize,
    kernel_size: (usize, usize, usize, usize),
    strides: usize,
    input_size: (usize, usize, usize),
    pub output_size: (usize, usize, usize), // (filters, height, width)
    kernels: Array4<f64>,
    #[serde(skip)]
    output: Array3<f64>,
    #[serde(skip)]
    input: Array3<f64>,
}

impl ConvolutionalLayer {
    pub fn new(
        filters: usize,
        kernel_size: usize,
        strides: usize,
        input_size: (usize, usize, usize),
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (kernel_size * kernel_size * input_size.0) as f64).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();

        let kernels =
            Array4::from_shape_fn((filters, input_size.0, kernel_size, kernel_size), |_| {
                rng.sample(normal_distr)
            });

        let output_size = (
            filters,
            (input_size.1 - kernel_size) / strides + 1,
            (input_size.2 - kernel_size) / strides + 1,
        );

        ConvolutionalLayer {
            filters,
            kernel_size: (filters, input_size.0, kernel_size, kernel_size),
            strides,
            input_size,
            kernels,
            output_size,
            output: Array3::zeros((0, 0, 0)),
            input: Array3::zeros((0, 0, 0)),
        }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array3<f64> {
        self.input = input.to_owned();
        self.output = Array3::zeros(self.output_size);
        let mut indexed_output_iter: Vec<((usize, usize, usize), &mut f64)> =
            self.output.indexed_iter_mut().collect();
        indexed_output_iter
            .par_iter_mut()
            .for_each(|((f, y, x), output_val)| {
                let kernel_slice = self.kernels.slice(s![*f, .., .., ..]);
                let input_slice = input.slice(s![
                    ..,
                    *y..*y + self.kernel_size.2,
                    *x..*x + self.kernel_size.3
                ]);

                **output_val = relu((&input_slice * &kernel_slice).sum());
            });
        self.output.clone()
    }

    pub fn backward_propagate(&mut self, mut error: Array3<f64>, lr: f64) -> Array3<f64> {
        error
            .iter_mut()
            .zip(self.output.iter())
            .par_bridge()
            .for_each(|(e, o)| *e *= relu_prime(*o));

        let mut next_error: Array3<f64> = Array3::zeros(self.input_size);
        next_error
            .axis_iter_mut(Axis(1))
            .zip(0..self.input_size.1)
            .par_bridge()
            .for_each(|(mut row, row_i)| {
                for col in 0..self.output_size.2 {
                    for filter in 0..self.kernel_size.0 {
                        for kernel_row in 0..self.kernel_size.2 {
                            if kernel_row > row_i
                                || self.kernel_size.2 - kernel_row + row_i > self.input_size.1
                            {
                                continue;
                            }

                            row.slice_mut(s![.., col..col + self.kernel_size.3])
                                .add_assign(
                                    &(error[[filter, row_i - kernel_row, col]]
                                        * &self.kernels.slice(s![filter, .., kernel_row, ..])),
                                );
                        }
                    }
                }
            });

        let mut delta_k = Array4::zeros((
            self.filters,
            self.input_size.0,
            self.kernel_size.2,
            self.kernel_size.3,
        ));
        delta_k
            .outer_iter_mut()
            .into_par_iter()
            .zip(0..self.kernel_size.0)
            .for_each(|(mut kernel, kernel_index)| {
                for row in 0..self.output_size.1 {
                    for col in 0..self.output_size.2 {
                        if self.output[[kernel_index, row, col]] <= 0.0 {
                            continue;
                        }
                        kernel.add_assign(
                            &(&self.input.slice(s![
                                ..,
                                row..row + self.kernel_size.2,
                                col..col + self.kernel_size.3
                            ]) * error[[kernel_index, row, col]]),
                        );
                    }
                }
            });
        self.kernels -= &(&delta_k * lr);

        next_error
    }
}
