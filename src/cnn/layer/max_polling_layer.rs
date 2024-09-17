use super::LayerTrait;
use ndarray::{Array3, Array4};

pub struct MaxPollingLayer {
    kernel_size: usize,
    strides: usize,
    input_size: (usize, usize, usize),
    pub output_size: (usize, usize, usize),
    max_indices: Array4<usize>,
}

impl MaxPollingLayer {
    pub fn new(kernel_size: usize, input_size: (usize, usize, usize), strides: usize) -> Self {
        let output_width: usize = ((input_size.1 - kernel_size) / strides) + 1;
        let output_size = (input_size.0, output_width, output_width);
        MaxPollingLayer {
            kernel_size,
            strides,
            input_size,
            output_size,
            max_indices: Array4::<usize>::zeros((output_width, output_width, input_size.2, 2)),
        }
    }
}

impl LayerTrait for MaxPollingLayer {
    // Need a fresh review
    fn forward_propogate(&mut self, input: Array3<f32>) -> Array3<f32> {
        let mut output: Array3<f32> = Array3::<f32>::zeros(self.output_size);

        for ((x, y, f), output_val) in output.indexed_iter_mut() {
            *output_val = -1.0;
            self.max_indices[[x, y, f, 0]] = 0;
            self.max_indices[[x, y, f, 1]] = 0;

            for ky in 0..self.kernel_size {
                for kx in 0..self.kernel_size {
                    let index: (usize, usize) = (x * self.strides + kx, y * self.strides + ky);
                    let value: f32 = input[[index.0, index.1, f]];

                    if value > *output_val {
                        *output_val = value;
                        self.max_indices[[x, y, f, 0]] = index.0;
                        self.max_indices[[x, y, f, 1]] = index.1;
                    }
                }
            }
        }
        output
    }

    fn backward_propogate(&mut self, error: Array3<f32>) -> Array3<f32> {
        // TODO: implement
        error
    }
}
