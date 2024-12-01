use ndarray::{aview0, Array, Array1, Axis, Dim, Dimension, RemoveAxis};
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
        BatchNormLayer {
            axis,
            beta: Array1::zeros(axis_size),
            epsilon,
            gamma: Array1::ones(axis_size),
            input_shape,
            momentum,
            moving_mean: Array1::zeros(axis_size),
            moving_variance: Array1::zeros(axis_size),
        }
    }

    pub fn input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    pub fn forward_propagate<D>(
        &mut self,
        mut activations: Array<f64, D>,
        is_training: bool,
    ) -> (Array<f64, D>, Option<BNCache<D>>)
    where
        D: Dimension,
        D: RemoveAxis,
    {
        if is_training {
            let mean = self.mean_across_axes(activations.clone(), self.axis);
            let xmu = activations - &mean;
            let var = self.mean_across_axes(xmu.powi(2), self.axis);
            let var_ep = &var + self.epsilon;
            let var_inv = 1. / var_ep.sqrt();
            let xhat = &xmu * &var_inv;

            let output = self.scale_and_shift(xhat.clone());

            self.update_moving_mean(&mean);
            self.update_moving_variance(&var);

            let cache = BNCache {
                xmu,
                var: var_ep,
                var_inv,
                xhat,
            };
            (output, Some(cache))
        } else {
            activations
                .axis_iter_mut(Axis(self.axis))
                .enumerate()
                .par_bridge()
                .for_each(|(idx, mut actvn)| {
                    actvn.assign(
                        &((&actvn - self.moving_mean[[idx]])
                            / (self.moving_variance[[idx]] + self.epsilon).sqrt()),
                    );
                });
            activations = self.scale_and_shift(activations);
            (activations, None)
        }
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
        let m = self.calculate_reduced_size(error.shape(), self.axis);

        let d_xhat: Array<f64, D> =
            self.mul_along_axis_with_1dim(error.clone(), &self.gamma, self.axis);

        let d_var: Array<f64, D> = &d_xhat * &cache.xmu * &(cache.var.powf(-3. / 2.) * -0.5);
        let d_var = self.sum_across_axes(d_var, self.axis);

        let d_mu = self.sum_across_axes(&d_xhat * &(-1. * &cache.var_inv), self.axis)
            + &d_var * self.mean_across_axes(-2. * &cache.xmu, self.axis);

        let d_x = d_xhat * &cache.var_inv + &d_var * 2. * &cache.xmu / m + &d_mu / m;

        let d_gamma = self.sum_across_axes(&error * cache.xhat, self.axis);
        self.update_gamma(&d_gamma, lr);

        let d_beta = self.sum_across_axes(error, self.axis);
        self.update_beta(&d_beta, lr);

        d_x
    }

    fn update_moving_mean<D>(&mut self, mean: &Array<f64, D>)
    where
        D: Dimension + RemoveAxis,
    {
        self.moving_mean
            .iter_mut()
            .zip(mean.axis_iter(Axis(self.axis)))
            .par_bridge()
            .for_each(|(mvng_mu, mu)| {
                *mvng_mu = *mvng_mu * self.momentum + mu.first().unwrap() * (1. - self.momentum)
            });
    }

    fn update_moving_variance<D>(&mut self, var: &Array<f64, D>)
    where
        D: Dimension + RemoveAxis,
    {
        self.moving_variance
            .iter_mut()
            .zip(var.axis_iter(Axis(self.axis)))
            .par_bridge()
            .for_each(|(mvng_var, var)| {
                *mvng_var = *mvng_var * self.momentum + var.first().unwrap() * (1. - self.momentum)
            });
    }

    fn update_beta<D>(&mut self, beta_grad: &Array<f64, D>, lr: f64)
    where
        D: Dimension + RemoveAxis,
    {
        self.beta
            .iter_mut()
            .zip(beta_grad.axis_iter(Axis(self.axis)))
            .par_bridge()
            .for_each(|(beta, beta_grd)| *beta -= beta_grd.first().unwrap() * lr);
    }

    fn update_gamma<D>(&mut self, gamma_grad: &Array<f64, D>, lr: f64)
    where
        D: Dimension + RemoveAxis,
    {
        self.gamma
            .iter_mut()
            .zip(gamma_grad.axis_iter(Axis(self.axis)))
            .par_bridge()
            .for_each(|(gamma, gamma_grd)| *gamma -= gamma_grd.first().unwrap() * lr);
    }

    fn scale_and_shift<D>(&self, mut activations: Array<f64, D>) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        activations
            .axis_iter_mut(Axis(self.axis))
            .enumerate()
            .par_bridge()
            .for_each(|(idx, mut out)| {
                out *= self.gamma[[idx]];
                out += self.beta[[idx]];
            });
        activations
    }

    fn calculate_reduced_size(&self, shape: &[usize], axis: usize) -> f64 {
        shape.iter().product::<usize>() as f64 / shape[axis] as f64
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

    fn mean_across_axes<D>(&self, mtrx: Array<f64, D>, axis: usize) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        let m = self.calculate_reduced_size(mtrx.shape(), axis);
        self.sum_across_axes(mtrx, axis) / aview0(&m)
    }

    fn sum_across_axes<D>(&self, mut mtrx: Array<f64, D>, axis: usize) -> Array<f64, D>
    where
        D: Dimension + RemoveAxis,
    {
        mtrx.axis_iter_mut(Axis(axis))
            .par_bridge()
            .for_each(|mut ax| {
                ax.fill(ax.sum());
            });
        mtrx
    }
}
