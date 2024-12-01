use crate::{
    cnn::optimizer::{Optimizer, OptimizerType},
    config::CONV_WEIGHT_SCALE,
};
use ndarray::{s, Array1, Array4, Array5, Axis, Dim, IntoDimension};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;
use std::ops::AddAssign;

type Tensor4DShape = (usize, usize, usize, usize);

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConvolutionalLayer {
    kernel_shape: Tensor4DShape,
    strides: usize,
    input_shape: Tensor4DShape,
    output_shape: Tensor4DShape, // (batch, filters, height, width)
    kernels: Array4<f64>,
    biases: Array1<f64>,
    k_optimizer: Optimizer<Dim<[usize; 4]>>,
    b_optimizer: Optimizer<Dim<[usize; 1]>>,
    use_bias: bool,
    #[serde(skip)]
    input: Array4<f64>,
    padding: Tensor4DShape, // (left, right, top, bottom)
}

impl ConvolutionalLayer {
    pub fn new(
        filters: usize,
        kernel_size: usize,
        strides: usize,
        mut input_shape: Tensor4DShape,
        bias: Option<f64>,
        padding: bool,
        optimizer_type: &OptimizerType,
    ) -> Self {
        if strides == 0 {
            panic!("Conv: Stride should be greater than 0.");
        }
        if input_shape.2 < kernel_size || input_shape.3 < kernel_size {
            panic!("Conv: Input shape can not be smaller than kernel.");
        }
        if kernel_size == 0 || filters == 0 {
            panic!("Conv: Incorrect kernel or filters size.");
        }

        let mut rng = rand::thread_rng();
        let std_dev = (CONV_WEIGHT_SCALE / (kernel_size * kernel_size * filters) as f64).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();
        let kernel_shape = (filters, input_shape.1, kernel_size, kernel_size);
        let kernels = Array4::from_shape_fn(kernel_shape, |_| rng.sample(normal_distr));

        let (use_bias, bias) = match bias {
            Some(bias) => (true, bias),
            None => (false, 0.),
        };
        let biases = Array1::from_elem(filters, bias);

        let mut output_shape = (
            input_shape.0,
            filters,
            (input_shape.2 - kernel_size) / strides + 1,
            (input_shape.3 - kernel_size) / strides + 1,
        );

        if padding {
            output_shape = (
                input_shape.0,
                filters,
                (input_shape.2 / strides).max(1),
                (input_shape.3 / strides).max(1),
            );
            input_shape = (
                input_shape.0,
                input_shape.1,
                input_shape.2 + kernel_size - 1,
                input_shape.3 + kernel_size - 1,
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
            kernel_shape,
            strides,
            input_shape,
            k_optimizer: optimizer_type.init(kernels.dim().into_dimension()),
            b_optimizer: optimizer_type.init(biases.dim().into_dimension()),
            kernels,
            output_shape,
            input: Array4::zeros((0, 0, 0, 0)),
            use_bias,
            biases,
            padding,
        }
    }

    pub fn forward_propagate(&mut self, input: Array4<f64>, _is_training: bool) -> Array4<f64> {
        self.input = self.prepare(input);
        self.calculate_output(&self.input)
    }

    pub fn backward_propagate(&mut self, error: Array4<f64>) -> Array4<f64> {
        let next_error = self.calculate_next_err(&error);
        self.kernels -= &self.k_optimizer.optimize(self.calculate_delta_k(&error));
        if self.use_bias {
            self.biases -= &self.b_optimizer.optimize(self.calculate_delta_b(&error));
        }
        next_error
    }

    pub fn output_shape(&self) -> Tensor4DShape {
        self.output_shape
    }

    fn prepare(&self, input: Array4<f64>) -> Array4<f64> {
        if self.padding == (0, 0, 0, 0) {
            return input;
        }

        let mut padded_input = Array4::zeros(self.input_shape);
        padded_input
            .slice_mut(s![
                ..,
                ..,
                self.padding.2..self.input_shape.2 - self.padding.3,
                self.padding.0..self.input_shape.3 - self.padding.1,
            ])
            .assign(&input);
        padded_input
    }

    fn calculate_output(&self, input: &Array4<f64>) -> Array4<f64> {
        let mut output: Array4<f64> = Array4::zeros(self.output_shape);
        output
            .outer_iter_mut()
            .zip(input.outer_iter())
            .par_bridge()
            .for_each(|(mut op, inp)| {
                op.indexed_iter_mut()
                    .par_bridge()
                    .for_each(|((f, mut y, mut x), output_val)| {
                        y *= self.strides;
                        x *= self.strides;
                        *output_val = (&inp.slice(s![
                            ..,
                            y..y + self.kernel_shape.2,
                            x..x + self.kernel_shape.3
                        ]) * &self.kernels.slice(s![f, .., .., ..]))
                            .sum()
                            + self.biases[[f]];
                    });
            });
        output
    }

    fn calculate_next_err(&self, err: &Array4<f64>) -> Array4<f64> {
        let mut next_error: Array4<f64> = Array4::zeros(self.input_shape);
        next_error
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(n_err_i, mut n_err)| {
                n_err
                    .axis_iter_mut(Axis(1))
                    .enumerate()
                    .par_bridge()
                    .for_each(|(row_i, mut row)| {
                        if row_i < self.padding.2 {
                            return;
                        }
                        for kernel_row in 0..self.kernel_shape.2 {
                            if kernel_row > row_i
                                || self.kernel_shape.2 - kernel_row + row_i > self.input_shape.2
                                || (row_i - kernel_row) as f64 % self.strides as f64 != 0.0
                            {
                                continue;
                            }

                            for col_i in (0..self.input_shape.3 - self.kernel_shape.3 + 1)
                                .step_by(self.strides)
                            {
                                for filter in 0..self.kernel_shape.0 {
                                    row.slice_mut(s![.., col_i..col_i + self.kernel_shape.3])
                                        .add_assign(
                                            &(&self.kernels.slice(s![filter, .., kernel_row, ..])
                                                * err[[
                                                    n_err_i,
                                                    filter,
                                                    (row_i - kernel_row) / self.strides,
                                                    col_i / self.strides,
                                                ]]),
                                        );
                                }
                            }
                        }
                    });
            });

        if self.padding != (0, 0, 0, 0) {
            let mut unpadded_err_grad: Array4<f64> = Array4::zeros((
                self.input_shape.0,
                self.input_shape.1,
                self.input_shape.2 - self.padding.2 - self.padding.3,
                self.input_shape.3 - self.padding.0 - self.padding.1,
            ));
            unpadded_err_grad.assign(&next_error.slice(s![
                ..,
                ..,
                self.padding.2..self.input_shape.2 - self.padding.3,
                self.padding.0..self.input_shape.3 - self.padding.1,
            ]));
            next_error = unpadded_err_grad;
        }
        next_error
    }

    fn calculate_delta_k(&self, err: &Array4<f64>) -> Array4<f64> {
        let mut kernel_grads: Array5<f64> = Array5::zeros((
            self.input_shape.0,
            self.kernel_shape.0,
            self.kernel_shape.1,
            self.kernel_shape.2,
            self.kernel_shape.3,
        ));
        kernel_grads
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(kg_i, mut kg)| {
                kg.outer_iter_mut()
                    .enumerate()
                    .par_bridge()
                    .for_each(|(kernel_i, mut kernel)| {
                        for row in
                            (0..self.input_shape.2 - self.kernel_shape.2 + 1).step_by(self.strides)
                        {
                            for col in (0..self.input_shape.3 - self.kernel_shape.3 + 1)
                                .step_by(self.strides)
                            {
                                kernel.add_assign(
                                    &(&self.input.slice(s![
                                        kg_i,
                                        ..,
                                        row..row + self.kernel_shape.2,
                                        col..col + self.kernel_shape.3
                                    ]) * err
                                        [[kg_i, kernel_i, row / self.strides, col / self.strides]]),
                                );
                            }
                        }
                    });
            });

        kernel_grads.mean_axis(Axis(0)).unwrap()
    }

    fn calculate_delta_b(&self, err: &Array4<f64>) -> Array1<f64> {
        err.mean_axis(Axis(0))
            .unwrap()
            .mean_axis(Axis(1))
            .unwrap()
            .mean_axis(Axis(1))
            .unwrap()
    }
}
