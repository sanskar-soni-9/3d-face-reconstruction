use ndarray::{Array1, Array3};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FlattenLayer {
    pub input_size: (usize, usize, usize),
}

impl FlattenLayer {
    pub fn new(input_size: (usize, usize, usize)) -> Self {
        FlattenLayer { input_size }
    }

    pub fn forward_propagate(&mut self, input: &Array3<f64>, _is_training: bool) -> Array1<f64> {
        input.flatten().to_owned()
    }

    pub fn backward_propagate(&mut self, error: &Array1<f64>) -> Array3<f64> {
        error
            .to_owned()
            .into_shape_with_order(self.input_size)
            .unwrap()
    }
}
