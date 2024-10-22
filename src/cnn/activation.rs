#[derive(serde::Serialize, serde::Deserialize)]
pub enum Activation {
    Linear,
    ReLU,
    ReLU6,
    Sigmoid,
    SiLU,
}

impl Activation {
    pub fn activate(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => Self::linear(x),
            Activation::ReLU => Self::relu(x),
            Activation::ReLU6 => Self::relu6(x),
            Activation::Sigmoid => Self::sigmoid(x),
            Activation::SiLU => Self::silu(x),
        }
    }
    pub fn deactivate(&self, x: f64) -> f64 {
        match self {
            Activation::Linear => Self::linear_deriv(x),
            Activation::ReLU => Self::relu_deriv(x),
            Activation::ReLU6 => Self::relu6_deriv(x),
            Activation::Sigmoid => Self::sigmoid_deriv(x),
            Activation::SiLU => Self::silu_deriv(x),
        }
    }

    pub fn linear(x: f64) -> f64 {
        x
    }

    pub fn linear_deriv(_: f64) -> f64 {
        1.0
    }

    pub fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    pub fn relu_deriv(x: f64) -> f64 {
        x.clamp(0.0, 1.0)
    }

    pub fn relu6(x: f64) -> f64 {
        x.clamp(0.0, 6.0)
    }

    pub fn relu6_deriv(x: f64) -> f64 {
        if x > 0.0 && x < 6.0 {
            1.0
        } else {
            0.0
        }
    }

    pub fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn sigmoid_deriv(x: f64) -> f64 {
        let sigmoid_x = Self::sigmoid(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }

    pub fn silu(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }

    pub fn silu_deriv(x: f64) -> f64 {
        let sigmoid_x = Self::sigmoid(x);
        sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))
    }
}
