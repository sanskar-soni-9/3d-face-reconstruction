use ndarray::Array3;

pub mod convolutional_layer;
pub mod max_polling_layer;

pub enum LayerType {
    ConvolutionalLayer(convolutional_layer::ConvolutionalLayer),
    MaxPollingLayer(max_polling_layer::MaxPollingLayer),
}

pub trait LayerTrait {
    fn forward_propagate(&mut self, input: &Array3<f32>) -> Array3<f32>;
    fn backward_propagate(&mut self, error: &Array3<f32>) -> Array3<f32>;
}
