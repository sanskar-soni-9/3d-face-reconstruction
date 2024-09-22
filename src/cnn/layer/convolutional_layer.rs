use ndarray::{s, Array3, Array4};
use rand::Rng;
use rand_distr::Normal;
use rayon::prelude::*;

pub struct ConvolutionalLayer {
    filters: usize,
    kernel_size: usize,
    strides: usize,
    input_size: (usize, usize, usize),
    pub output_size: (usize, usize, usize), // (filters, width, height)
    kernels: Array4<f32>,
    output: Array3<f32>,
}

impl ConvolutionalLayer {
    pub fn new(
        filters: usize,
        kernel_size: usize,
        strides: usize,
        input_size: (usize, usize, usize),
    ) -> Self {
        let mut rng = rand::thread_rng();
        let std_dev = (2.0 / (kernel_size * kernel_size * input_size.0) as f32).sqrt();
        let normal_distr = Normal::new(0.0, std_dev).unwrap();

        let kernels =
            Array4::from_shape_fn([filters, input_size.0, kernel_size, kernel_size], |_| {
                rng.sample(normal_distr)
            });

        let output_size = (
            filters,
            (input_size.1 - kernel_size) / strides + 1,
            (input_size.2 - kernel_size) / strides + 1,
        );

        ConvolutionalLayer {
            filters,
            kernel_size,
            strides,
            input_size,
            kernels,
            output_size,
            output: Array3::zeros(output_size),
        }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f32>, is_training: bool) -> Array3<f32> {
        let mut var: Vec<((usize, usize, usize), &mut f32)> =
            self.output.indexed_iter_mut().collect();
        var.par_iter_mut()
            .enumerate()
            .for_each(|(_, ((f, x, y), output_val))| {
                let kernel_slice = self.kernels.slice(s![*f, .., .., ..]);
                let input_slice =
                    input.slice(s![.., *x..*x + self.kernel_size, *y..*y + self.kernel_size]);

                **output_val = (&input_slice * &kernel_slice).sum().max(0.0);
            });
        self.output.clone()
    }

    pub fn backward_propagate(&mut self, error: &Array3<f32>) -> Array3<f32> {
        // TODO: implement
        error.clone()
    }
}
