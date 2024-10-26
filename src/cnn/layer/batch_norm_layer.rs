use ndarray::{Array, Array1, Axis, Dim, Dimension, RemoveAxis};
use rayon::iter::{ParallelBridge, ParallelIterator};

#[derive(serde::Serialize, serde::Deserialize)]
pub struct BatchNormLayer {
    axis: usize,
    beta: Array1<f64>,
    epsilon: f64,
    gamma: Array1<f64>,
    input_shape: Vec<usize>,
    momentum: f64,
    moving_mean: Array1<f64>,
    moving_variance: Array1<f64>,
    reduced_axes: Vec<usize>,
}

pub struct BNCache<D>
where
    D: Dimension,
{
    xmu: Array<f64, D>,
    var: Array<f64, D>,
    var_inv: Array<f64, D>,
    xhat: Array<f64, D>,
}

impl BatchNormLayer {
    pub fn new(axis: usize, epsilon: f64, input_shape: Vec<usize>, momentum: f64) -> Self {
        let axis_size = input_shape[axis];
        let reduced_axes: Vec<usize> = (0..input_shape.len()).filter(|&x| x != axis).collect();
        BatchNormLayer {
            axis,
            beta: Array1::zeros(axis_size),
            epsilon,
            gamma: Array1::ones(axis_size),
            input_shape,
            momentum,
            moving_mean: Array1::zeros(axis_size),
            moving_variance: Array1::zeros(axis_size),
            reduced_axes,
        }
    }

    pub fn forward_propagate<D>(
        &mut self,
        activations: Array<f64, D>,
        is_training: bool,
    ) -> (Array<f64, D>, Option<BNCache<D>>)
    where
        D: Dimension,
        D: RemoveAxis,
    {
        // TODO: Implement Infer & Update Moving avgs
        if is_training {
            let mean = self.mean_reduced_axes(activations.clone(), &self.reduced_axes);
            let xmu = activations - mean;
            let var = self.mean_reduced_axes(xmu.powi(2), &self.reduced_axes) + self.epsilon;
            let var_inv = 1. / var.sqrt();
            let xhat = &xmu * &var_inv;
            let mut output = xhat.clone();
            output
                .axis_iter_mut(Axis(self.axis))
                .zip(self.gamma.iter())
                .zip(self.beta.iter())
                .for_each(|((mut out, gm), bt)| {
                    out *= *gm;
                    out += *bt;
                });

            let cache = BNCache {
                xmu,
                var,
                var_inv,
                xhat,
            };

            return (output, Some(cache));
        }
        (activations, None)
    }

    pub fn backward_propagate<D>(
        &mut self,
        error: Array<f64, D>,
        cache: BNCache<D>,
        lr: f64,
    ) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        let d_beta = error.sum_axis(Axis(self.axis));

        let d_gamma: Array<f64, D> = &error * &cache.xhat;
        let d_gamma = d_gamma.sum_axis(Axis(self.axis));

        let m = self.get_reduction_size(error.shape(), &self.reduced_axes) as f64;

        let d_xhat: Array<f64, D> = self.mul_along_axis_with_1dim(error, &self.gamma, self.axis);

        let d_var: Array<f64, D> = &d_xhat * &cache.xmu * &(cache.var.powf(-3. / 2.) * -0.5);
        let d_var = self.sum_reduced_axes(d_var, &self.reduced_axes);

        let d_mu = self.sum_reduced_axes(&d_xhat * &(-1. * &cache.var_inv), &self.reduced_axes)
            + &d_var * self.mean_reduced_axes(-2. * &cache.xmu, &self.reduced_axes);

        let d_x = d_xhat * &cache.var_inv + &d_var * 2. * &cache.xmu / m + &d_mu / m;

        self.update_beta::<D>(&d_beta, lr);
        self.update_gamma::<D>(&d_gamma, lr);

        d_x
    }

    fn update_beta<D>(&mut self, beta_grad: &Array<f64, <D as Dimension>::Smaller>, lr: f64)
    where
        D: Dimension,
    {
        self.beta
            .iter_mut()
            .zip(beta_grad.iter())
            .par_bridge()
            .for_each(|(beta, beta_grd)| *beta -= beta_grd * lr);
    }

    fn update_gamma<D>(&mut self, gamma_grad: &Array<f64, <D as Dimension>::Smaller>, lr: f64)
    where
        D: Dimension,
    {
        self.gamma
            .iter_mut()
            .zip(gamma_grad.iter())
            .par_bridge()
            .for_each(|(gamma, gamma_grd)| *gamma -= gamma_grd * lr);
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn mul_along_axis_with_1dim<D>(
        &self,
        mut m1: Array<f64, D>,
        m2: &Array<f64, Dim<[usize; 1]>>,
        axis: usize,
    ) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        m1.axis_iter_mut(Axis(axis))
            .zip(m2)
            .par_bridge()
            .for_each(|(mut ax, m)| ax *= *m);
        m1
    }

    fn get_reduction_size(&self, shape: &[usize], reduction_axes: &[usize]) -> usize {
        reduction_axes.iter().map(|ax| shape[*ax]).product()
    }

    fn mean_reduced_axes<D>(&self, mtrx: Array<f64, D>, reduction_axes: &[usize]) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        let total_elements: usize = self.get_reduction_size(mtrx.shape(), reduction_axes);
        self.sum_reduced_axes(mtrx, reduction_axes) / total_elements as f64
    }

    fn sum_reduced_axes<D>(
        &self,
        mut mtrx: Array<f64, D>,
        reduction_axes: &[usize],
    ) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        for &axis in reduction_axes.iter() {
            let sum = mtrx.sum_axis(Axis(axis)).insert_axis(Axis(axis));
            mtrx = sum.broadcast(mtrx.raw_dim()).unwrap().to_owned();
        }
        mtrx
    }
}
