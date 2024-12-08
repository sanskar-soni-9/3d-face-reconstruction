use crate::utils::elementwise_max;
use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AdamParameters {
    pub epsilon: f64,
    pub beta_1: f64,
    pub beta_2: f64,
    pub ams_grad: bool,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Adam<D>
where
    D: Dimension,
{
    epsilon: f64,
    beta_1: f64,
    beta_2: f64,
    momentum: Array<f64, D>,
    velocity: Array<f64, D>,
    max_velocity: Option<Array<f64, D>>,
    iteration_num: usize,
}

impl<D> Adam<D>
where
    D: Dimension,
{
    pub fn new(params: &AdamParameters, shape: D) -> Self {
        let max_velocity = if params.ams_grad {
            Some(Array::zeros(shape.clone()))
        } else {
            None
        };
        Self {
            epsilon: params.epsilon,
            beta_1: params.beta_1,
            beta_2: params.beta_2,
            momentum: Array::zeros(shape.clone()),
            velocity: Array::zeros(shape),
            max_velocity,
            iteration_num: 0,
        }
    }

    pub fn optimize(&mut self, changes: Array<f64, D>, lr: f64) -> Array<f64, D>
    where
        D: Dimension,
    {
        self.iteration_num += 1;
        self.momentum = &self.momentum * self.beta_1 + (1. - self.beta_1) * &changes;
        self.velocity = &self.velocity * self.beta_2 + (1. - self.beta_2) * changes.pow2();

        let m_hat = &self.momentum / (1. - self.beta_1.powi(self.iteration_num as i32));
        let v_hat = &self.velocity / (1. - self.beta_2.powi(self.iteration_num as i32));
        let v = if let Some(max_velocity) = &mut self.max_velocity {
            *max_velocity = elementwise_max(max_velocity, &v_hat);
            max_velocity
        } else {
            &v_hat
        };

        m_hat * lr / (v.sqrt() + self.epsilon)
    }
}
