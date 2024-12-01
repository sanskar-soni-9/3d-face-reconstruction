use crate::config::{DEFAULT_LEARNING_RATE, SGD_MOMENTUM};
use ndarray::{Array, Dimension};
pub use sgd_momentum::SGDMParameters;

mod sgd_momentum;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum OptimizerType {
    SGDMomentum(SGDMParameters),
}

impl Default for OptimizerType {
    fn default() -> Self {
        Self::SGDMomentum(SGDMParameters {
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
            Self::SGDMomentum(sgd_params) => {
                Optimizer::SGDMomentum(sgd_momentum::SGDMomentum::new(sgd_params, shape))
            }
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum Optimizer<D>
where
    D: Dimension,
{
    SGDMomentum(sgd_momentum::SGDMomentum<D>),
}

impl<D> Optimizer<D>
where
    D: Dimension,
{
    pub fn optimize(&mut self, changes: Array<f64, D>) -> &Array<f64, D> {
        match self {
            Optimizer::SGDMomentum(optimizer) => optimizer.optimize(changes),
        }
    }
}
