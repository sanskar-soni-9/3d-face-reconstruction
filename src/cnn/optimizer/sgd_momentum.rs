use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SgdmParameters {
    pub momentum: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct SgdMomentum<D>
where
    D: Dimension,
{
    momentum: f64,
    changes: Array<f64, D>,
}

impl<D> SgdMomentum<D>
where
    D: Dimension,
{
    pub fn new(params: &SgdmParameters, shape: D) -> Self {
        Self {
            momentum: params.momentum,
            changes: Array::default(shape),
        }
    }

    pub fn optimize(&mut self, changes: Array<f64, D>, lr: f64) -> Array<f64, D> {
        self.changes = changes * lr + &self.changes * self.momentum;
        self.changes.clone()
    }
}
