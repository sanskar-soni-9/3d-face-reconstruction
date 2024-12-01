use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SgdParameters {
    pub lr: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Sgd {
    lr: f64,
}

impl Sgd {
    pub fn new(params: &SgdParameters) -> Self {
        Self { lr: params.lr }
    }

    pub fn optimize<D>(&mut self, changes: Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        changes * self.lr
    }
}
