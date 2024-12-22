use ndarray::{Array, Dimension};

pub struct OperandCache<D>
where
    D: Dimension,
{
    pub main_actvns: Array<f64, D>, // Main branch
    pub skip_actvns: Array<f64, D>, // Skip branch
}

impl<D> OperandCache<D>
where
    D: Dimension,
{
    pub fn update_skip(&mut self, skip_actvns: Array<f64, D>) {
        self.skip_actvns = skip_actvns;
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub enum OperationType {
    Add,
    Mul,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct OperandLayer {
    id: usize,
    input_shape: Vec<usize>,
}

impl OperandLayer {
    pub fn new(id: usize, input_shape: Vec<usize>) -> Self {
        OperandLayer { id, input_shape }
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.input_shape
    }
    pub fn id(&self) -> usize {
        self.id
    }

    pub fn forward_propagate<D>(
        &self,
        input: Array<f64, D>,
        _is_training: bool,
    ) -> (Array<f64, D>, OperandCache<D>)
    where
        D: Dimension,
    {
        (
            input.clone(),
            OperandCache {
                main_actvns: Array::default(input.raw_dim()),
                skip_actvns: input,
            },
        )
    }

    pub fn backward_propagate<D>(
        &self,
        error_a: Array<f64, D>,
        cache: OperandCache<D>,
    ) -> Array<f64, D>
    where
        D: Dimension,
    {
        &error_a + &cache.skip_actvns
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct OperationLayer {
    input_shape: Vec<usize>,
    operand_id: usize,
    operation: OperationType,
}

impl OperationLayer {
    pub fn new(operand_id: usize, input_shape: Vec<usize>, operation: OperationType) -> Self {
        Self {
            input_shape,
            operand_id,
            operation,
        }
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn output_shape(&self) -> &[usize] {
        &self.input_shape
    }
    pub fn operand_id(&self) -> usize {
        self.operand_id
    }

    pub fn forward_propagate<D>(
        &self,
        input: Array<f64, D>,
        mut cache: OperandCache<D>,
        _is_training: bool,
    ) -> (Array<f64, D>, OperandCache<D>)
    where
        D: Dimension,
    {
        cache.main_actvns = input
            .broadcast(cache.skip_actvns.raw_dim())
            .expect("Operands should have same shape in OperationLayer")
            .to_owned();
        match self.operation {
            OperationType::Add => (self.add(&cache.main_actvns, &cache.skip_actvns), cache),
            OperationType::Mul => (self.multiply(&cache.main_actvns, &cache.skip_actvns), cache),
        }
    }

    pub fn backward_propagate<D>(
        &self,
        error: Array<f64, D>,
        mut cache: OperandCache<D>,
    ) -> (Array<f64, D>, OperandCache<D>)
    where
        D: Dimension,
    {
        match self.operation {
            OperationType::Add => {
                cache.skip_actvns = error.clone();
                (error, cache)
            }
            OperationType::Mul => {
                let next_error = &error * &cache.skip_actvns;
                cache.skip_actvns = &error * &cache.main_actvns;
                (next_error, cache)
            }
        }
    }

    fn add<D>(&self, a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        a + b
    }

    fn multiply<D>(&self, a: &Array<f64, D>, b: &Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension,
    {
        a * b
    }
}
