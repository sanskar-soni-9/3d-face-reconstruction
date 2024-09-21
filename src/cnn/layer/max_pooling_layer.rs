use ndarray::Array3;

pub struct MaxPoolingLayer {
    kernel_size: usize,
    strides: usize,
    input_size: (usize, usize, usize),
    pub output_size: (usize, usize, usize),
}

impl MaxPoolingLayer {
    pub fn new(kernel_size: usize, input_size: (usize, usize, usize), strides: usize) -> Self {
        let output_size = (
            input_size.0,
            (input_size.1 - kernel_size) / strides + 1,
            (input_size.2 - kernel_size) / strides + 1,
        );

        MaxPoolingLayer {
            kernel_size,
            strides,
            input_size,
            output_size,
        }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f32>) -> Array3<f32> {
        let mut output: Array3<f32> = Array3::<f32>::zeros(self.output_size);

        for ((f, x, y), output_val) in output.indexed_iter_mut() {
            for ky in 0..self.kernel_size {
                for kx in 0..self.kernel_size {
                    let index: (usize, usize) = (x * self.strides + kx, y * self.strides + ky);
                    let value: f32 = input[[f, index.0, index.1]];

                    if value > *output_val {
                        *output_val = value;
                    }
                }
            }
        }
        output
    }

    pub fn backward_propagate(&mut self, error: &Array3<f32>) -> Array3<f32> {
        // TODO: implement
        error.clone()
    }
}
