use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SGDMParameters {
    pub lr: f64,
    pub momentum: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SGDMomentum<D>
where
    D: Dimension,
{
    lr: f64,
    momentum: f64,
    changes: Array<f64, D>,
}

impl<D> SGDMomentum<D>
where
    D: Dimension,
{
    pub fn new(params: &SGDMParameters, shape: D) -> Self {
        Self {
            lr: params.lr,
            momentum: params.momentum,
            changes: Array::default(shape),
        }
    }

    pub fn optimize(&mut self, changes: Array<f64, D>) -> &Array<f64, D> {
        self.changes = changes * self.lr + &self.changes * self.momentum;
        &self.changes
    }
}
