use ndarray::{Array1, Array3};

pub struct FlattenLayer {
    pub input_size: (usize, usize, usize),
}

impl FlattenLayer {
    pub fn new(input_size: (usize, usize, usize)) -> Self {
        FlattenLayer { input_size }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f32>, is_training: bool) -> Array1<f32> {
        input.flatten().to_owned()
    }

    pub fn backward_propagate(&mut self, error: &Vec<f32>) -> Vec<f32> {
        error.to_owned()
    }
}
