use ndarray::Array3;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

type MaxIndices = Vec<(usize, usize, usize, usize, usize)>; // (channels, output_x, output_y, input_x, input_y)

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MaxPoolingLayer {
    kernel_size: usize,
    strides: usize,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    #[serde(skip)]
    max_indices: Vec<MaxIndices>,
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

    pub fn forward_propagate(
        &mut self,
        input: &[Array3<f64>],
        _is_training: bool,
    ) -> Vec<Array3<f64>> {
        self.max_indices.clear();
        let mut output: Vec<Array3<f64>> = vec![];
        input
            .iter()
            .for_each(|_| output.push(Array3::zeros(self.output_shape)));

        output
            .iter_mut()
            .zip(0..input.len())
            .for_each(|(output_arr, output_i)| {
                self.calculate_output(output_i, &input[output_i], output_arr)
            });

        output
    }

    pub fn backward_propagate(&self, error: Vec<Array3<f64>>) -> Vec<Array3<f64>> {
        let mut next_error: Vec<Array3<f64>> = vec![];
        error
            .iter()
            .for_each(|_| next_error.push(Array3::zeros(self.input_shape)));

        next_error
            .par_iter_mut()
            .zip(0..error.len())
            .for_each(|(n_err, i)| {
                for (f, x, y, ix, iy) in &self.max_indices[i] {
                    n_err[[*f, *iy, *ix]] = error[i][[*f, *y, *x]];
                }
            });
        next_error
    }

    fn calculate_output(&mut self, index: usize, input: &Array3<f64>, output: &mut Array3<f64>) {
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
            self.max_indices[index].push(max_index);
        }
    }

    pub fn output_shape(&self) -> (usize, usize, usize) {
        self.output_shape
    }
}
