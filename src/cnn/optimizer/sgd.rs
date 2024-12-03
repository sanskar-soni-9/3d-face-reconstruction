use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize, Default)]
pub struct Sgd {}

impl Sgd {
    pub fn new() -> Self {
        Sgd::default()
    }

    pub fn optimize<D>(&mut self, changes: Array<f64, D>, lr: f64) -> Array<f64, D>
    where
        D: Dimension,
    {
        changes * lr
    }
}
