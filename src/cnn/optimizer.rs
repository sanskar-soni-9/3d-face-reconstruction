use crate::config::{DEFAULT_LEARNING_RATE, SGD_MOMENTUM};
pub use adam::AdamParameters;
use ndarray::{Array, Dimension};
pub use sgd::SgdParameters;
pub use sgd_momentum::SgdmParameters;

mod adam;
mod sgd;
mod sgd_momentum;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum OptimizerType {
    Adam(AdamParameters),
    Sgd(SgdParameters),
    SgdMomentum(SgdmParameters),
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::SgdMomentum(SgdmParameters {
            lr: DEFAULT_LEARNING_RATE,
            momentum: SGD_MOMENTUM,
        })
    }
}

impl OptimizerType {
    pub fn init<D>(&self, shape: D) -> Optimizer<D>
    where
        D: Dimension,
    {
        match self {
            Self::Adam(adam_params) => Optimizer::Adam(adam::Adam::new(adam_params, shape)),
            Self::Sgd(sgd_params) => Optimizer::Sgd(sgd::Sgd::new(sgd_params)),
            Self::SgdMomentum(sgdm_params) => {
                Optimizer::SgdMomentum(sgd_momentum::SgdMomentum::new(sgdm_params, shape))
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum Optimizer<D>
where
    D: Dimension,
{
    Adam(adam::Adam<D>),
    Sgd(sgd::Sgd),
    SgdMomentum(sgd_momentum::SgdMomentum<D>),
}

impl<D> Optimizer<D>
where
    D: Dimension,
{
    pub fn optimize(&mut self, changes: Array<f64, D>) -> Array<f64, D> {
        match self {
            Self::Adam(optimizer) => optimizer.optimize(changes),
            Self::Sgd(optimizer) => optimizer.optimize(changes),
            Self::SgdMomentum(optimizer) => optimizer.optimize(changes),
        }
    }
}
