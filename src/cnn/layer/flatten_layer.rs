use ndarray::{Array1, Array3};

pub struct FlattenLayer {}

impl FlattenLayer {
    pub fn new() -> Self {
        FlattenLayer {}
    }

    pub fn forward_propagate(&mut self, input: &Array3<f32>) -> Array1<f32> {
        input.flatten().to_owned()
    }

    pub fn backward_propagate(&mut self, error: &Array3<f32>) -> Array3<f32> {
        // TODO: implement
        error.clone()
    }
}
