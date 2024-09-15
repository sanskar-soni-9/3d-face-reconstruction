use ndarray::Array3;

pub mod convolutional_layer;
pub mod max_polling_layer;

pub enum LayerType {
    ConvolutionalLayer(convolutional_layer::ConvolutionalLayer),
    MaxPollingLayer(max_polling_layer::MaxPollingLayer),
}

pub trait LayerTrait {
    fn forward_propogate(&mut self, input: Array3<f32>) -> Array3<f32>;
    fn backward_propogate(&mut self, error: Array3<f32>) -> Array3<f32>;
}
