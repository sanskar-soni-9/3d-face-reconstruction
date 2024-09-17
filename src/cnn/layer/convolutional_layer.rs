use super::LayerTrait;
use ndarray::{s, Array3, Array4};
use rand::{distributions::Uniform, Rng};

pub struct ConvolutionalLayer {
    filters: usize,
    kernel_size: usize,
    strides: usize,
    input_size: (usize, usize, usize),
    pub output_size: (usize, usize, usize), // (filters, width, height)
    kernels: Array4<f32>,
    input: Array3<f32>,
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
        let std_dev = (2.0 / input_size.2.pow(3) as f32).sqrt();
        let uniform = Uniform::new(-std_dev, std_dev);

        println!("stdDev: {}", std_dev);

        let kernels =
            Array4::from_shape_fn([filters, kernel_size, kernel_size, input_size.0], |_| {
                rng.sample(uniform)
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
            input: Array3::zeros(input_size),
            output: Array3::zeros(output_size),
        }
    }
}

impl LayerTrait for ConvolutionalLayer {
    fn forward_propogate(&mut self, input: Array3<f32>) -> Array3<f32> {
        self.input = input;
        for f in 0..self.output_size.0 {
            let kernel_slice = self.kernels.slice(s![f, .., .., ..]);
            for y in (0..self.output_size.2).step_by(self.strides) {
                for x in (0..self.output_size.1).step_by(self.strides) {
                    let input_slice =
                        self.input
                            .slice(s![x..x + self.kernel_size, y..y + self.kernel_size, ..]);
                    self.output[[x, y, f]] = (&input_slice * &kernel_slice).sum();
                }
            }
        }

        self.output.clone()
    }

    fn backward_propogate(&mut self, error: Array3<f32>) -> Array3<f32> {
        // TODO: implement
        error
    }
}
