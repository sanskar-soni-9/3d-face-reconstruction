use crate::utils::elementwise_max;
use ndarray::{Array, Dimension};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct AdamParameters {
    pub lr: f64,
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
    lr: f64,
    epsilon: f64,
    beta_1: f64,
    beta_2: f64,
    momentum: Array<f64, D>,
    velocity: Array<f64, D>,
    max_velocity: Array<f64, D>,
    iteration_num: usize,
    ams_grad: bool,
}

impl<D> Adam<D>
where
    D: Dimension,
{
    pub fn new(params: &AdamParameters, shape: D) -> Self {
        Self {
            lr: params.lr,
            epsilon: params.epsilon,
            beta_1: params.beta_1,
            beta_2: params.beta_2,
            momentum: Array::zeros(shape.clone()),
            velocity: Array::zeros(shape.clone()),
            max_velocity: Array::zeros(shape),
            iteration_num: 0,
            ams_grad: params.ams_grad,
        }
    }

    pub fn optimize(&mut self, changes: Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        self.iteration_num += 1;
        self.momentum = &self.momentum * self.beta_1 + (1. - self.beta_1) * &changes;
        self.velocity = &self.velocity * self.beta_2 + (1. - self.beta_2) * changes.pow2();

        let m_hat = &self.momentum / (1. - self.beta_1.powi(self.iteration_num as i32));
        let v_hat = &self.velocity / (1. - self.beta_2.powi(self.iteration_num as i32));
        let v = if self.ams_grad {
            self.max_velocity = elementwise_max(&self.max_velocity, &v_hat);
            &self.max_velocity
        } else {
            &v_hat
        };

        m_hat * self.lr / (v.sqrt() + self.epsilon)
    }
}
