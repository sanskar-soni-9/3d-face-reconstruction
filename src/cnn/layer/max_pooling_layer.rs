use ndarray::{s, Array4};
use rayon::iter::{ParallelBridge, ParallelIterator};

type MaxIndices = Vec<(usize, usize, usize, usize, usize)>; // (channels, output_x, output_y, input_x, input_y)

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MaxPoolingLayer {
    kernel_size: usize,
    strides: usize,
    input_shape: (usize, usize, usize, usize),
    output_shape: (usize, usize, usize, usize),
    #[serde(skip)]
    max_indices: Vec<MaxIndices>,
    padding: (usize, usize, usize, usize),
}

impl MaxPoolingLayer {
    pub fn new(
        kernel_size: usize,
        mut input_shape: (usize, usize, usize, usize),
        strides: usize,
        padding: bool,
    ) -> Self {
        let mut output_shape = (
            input_shape.0,
            input_shape.1,
            (input_shape.2 - kernel_size) / strides + 1,
            (input_shape.3 - kernel_size) / strides + 1,
        );
        if padding {
            output_shape = (
                input_shape.0,
                input_shape.1,
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

        MaxPoolingLayer {
            kernel_size,
            strides,
            input_shape,
            output_shape,
            max_indices: vec![],
            padding,
        }
    }

    pub fn forward_propagate(&mut self, input: Array4<f64>, _is_training: bool) -> Array4<f64> {
        self.calculate_output(self.prepare(input))
    }

    pub fn backward_propagate(&self, error: Array4<f64>) -> Array4<f64> {
        self.calculate_next_err(&error)
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

    fn calculate_output(&mut self, input: Array4<f64>) -> Array4<f64> {
        self.max_indices.clear();
        let mut output = Array4::zeros(self.output_shape);
        output
            .outer_iter_mut()
            .enumerate()
            .for_each(|(idx, mut op)| {
                self.max_indices.push(Vec::with_capacity(
                    self.output_shape.2 * self.output_shape.3,
                ));
                for ((f, y, x), output_val) in op.indexed_iter_mut() {
                    let mut max_index = (f, x, y, x, y);
                    for ky in 0..self.kernel_size {
                        for kx in 0..self.kernel_size {
                            let index: (usize, usize) =
                                (x * self.strides + kx, y * self.strides + ky);
                            let value: f64 = input[[idx, f, index.1, index.0]];

                            if value > *output_val {
                                *output_val = value;
                                max_index = (f, x, y, index.0, index.1);
                            }
                        }
                    }
                    self.max_indices[idx].push(max_index);
                }
            });
        output
    }

    fn calculate_next_err(&self, err: &Array4<f64>) -> Array4<f64> {
        let mut next_error: Array4<f64> = Array4::zeros(self.input_shape);
        next_error
            .outer_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(idx, mut n_err)| {
                for (f, x, y, ix, iy) in &self.max_indices[idx] {
                    n_err[[*f, *iy, *ix]] = err[[idx, *f, *y, *x]];
                }
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

    pub fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.output_shape
    }
}
