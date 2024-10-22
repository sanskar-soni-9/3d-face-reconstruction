use ndarray::Array3;

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MaxPoolingLayer {
    kernel_size: usize,
    strides: usize,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    #[serde(skip)]
    max_indices: Vec<(usize, usize, usize, usize, usize)>, // (channels, output_x, output_y, input_x, input_y)
}

impl MaxPoolingLayer {
    pub fn new(kernel_size: usize, input_shape: (usize, usize, usize), strides: usize) -> Self {
        let output_shape = (
            input_shape.0,
            (input_shape.1 - kernel_size) / strides + 1,
            (input_shape.2 - kernel_size) / strides + 1,
        );

        MaxPoolingLayer {
            kernel_size,
            strides,
            input_shape,
            output_shape,
            max_indices: vec![],
        }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array3<f64> {
        let mut output: Array3<f64> = Array3::zeros(self.output_shape);
        self.max_indices.clear();
        for ((f, y, x), output_val) in output.indexed_iter_mut() {
            let mut max_index = (f, x, y, x, y);
            for ky in 0..self.kernel_size {
                for kx in 0..self.kernel_size {
                    let index: (usize, usize) = (x * self.strides + kx, y * self.strides + ky);
                    let value: f64 = input[[f, index.1, index.0]];

                    if value > *output_val {
                        *output_val = value;
                        max_index = (f, x, y, index.0, index.1);
                    }
                }
            }
            self.max_indices.push(max_index);
        }
        output
    }

    pub fn backward_propagate(&self, error: Array3<f64>) -> Array3<f64> {
        let mut next_error: Array3<f64> = Array3::zeros(self.input_shape);
        for (f, x, y, ix, iy) in &self.max_indices {
            next_error[[*f, *iy, *ix]] = error[[*f, *y, *x]];
        }
        next_error
    }

    pub fn output_shape(&self) -> (usize, usize, usize) {
        self.output_shape
    }
}
