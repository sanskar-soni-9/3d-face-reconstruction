use ndarray::{Array2, Array4, Axis};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct ReshapeLayer {
    broadcast: bool,
    input_shape: (usize, usize),
    target_shape: (usize, usize, usize, usize),
}

impl ReshapeLayer {
    pub fn new(
        input_shape: (usize, usize),
        target_shape: (usize, usize, usize, usize),
        broadcast: bool,
    ) -> Self {
        Self {
            input_shape,
            target_shape,
            broadcast,
        }
    }

    pub fn forward_propagate(&self, input: &Array2<f64>, _is_training: bool) -> Array4<f64> {
        let output = input.to_owned().insert_axis(Axis(2)).insert_axis(Axis(3));
        if self.broadcast {
            output.broadcast(self.target_shape).unwrap().to_owned()
        } else {
            output
        }
    }

    pub fn backward_propagate(&self, error: &Array4<f64>, _lr: f64) -> Array2<f64> {
        error.to_owned().remove_axis(Axis(3)).remove_axis(Axis(2))
    }

    pub fn output_shape(&self) -> (usize, usize, usize, usize) {
        self.target_shape
    }
}
