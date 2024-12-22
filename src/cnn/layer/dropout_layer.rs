use ndarray::{Array, Array1, Dimension};
use rand::Rng;
use rayon::iter::{ParallelBridge, ParallelIterator};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct DropoutLayer {
    input_shape: Vec<usize>,
    mask: Array1<f64>,
    rate: f64,
}

impl DropoutLayer {
    pub fn new(input_shape: Vec<usize>, rate: f64) -> Self {
        Self {
            input_shape,
            mask: Array1::zeros(0),
            rate,
        }
    }

    pub fn forward_propagate<D>(
        &mut self,
        mut input: Array<f64, D>,
        _is_training: bool,
    ) -> Array<f64, D>
    where
        D: Dimension,
    {
        let droput_len: usize = input.shape().iter().product();
        let mut rng = rand::thread_rng();
        self.mask = Array1::from_shape_fn(droput_len, |_| {
            if rng.gen::<f64>() < self.rate {
                0.0
            } else {
                1.0 / (1.0 - self.rate) // Inverted Dropout
            }
        });
        input
            .iter_mut()
            .zip(&self.mask)
            .par_bridge()
            .for_each(|(inp, drpt)| *inp *= *drpt);
        input
    }

    pub fn backward_propagate<D>(&self, mut error: Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        error
            .iter_mut()
            .zip(&self.mask)
            .par_bridge()
            .for_each(|(err, drpt)| *err *= *drpt);
        error
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.input_shape
    }
}
