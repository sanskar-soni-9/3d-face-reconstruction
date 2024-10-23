use ndarray::{Array1, Array3};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct FlattenLayer {
    input_shape: (usize, usize, usize),
}

impl FlattenLayer {
    pub fn new(input_shape: (usize, usize, usize)) -> Self {
        FlattenLayer { input_shape }
    }

    pub fn forward_propagate(
        &mut self,
        input: &Vec<Array3<f64>>,
        _is_training: bool,
    ) -> Vec<Array1<f64>> {
        let mut output: Vec<Array1<f64>> = vec![];
        input
            .iter()
            .for_each(|inp| output.push(inp.flatten().to_owned()));
        output
    }

    pub fn backward_propagate(&self, error: &Vec<Array1<f64>>) -> Vec<Array3<f64>> {
        let mut next_error: Vec<Array3<f64>> = vec![];
        error.iter().for_each(|err| {
            next_error.push(
                err.to_owned()
                    .into_shape_with_order(self.input_shape)
                    .unwrap(),
            )
        });
        next_error
    }

    pub fn input_shape(&self) -> (usize, usize, usize) {
        self.input_shape
    }

    pub fn output_size(&self) -> usize {
        self.input_shape.0 * self.input_shape.1 * self.input_shape.2
    }
}
