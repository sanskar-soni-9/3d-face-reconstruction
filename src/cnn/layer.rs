pub mod convolutional_layer;
pub mod dense_layer;
pub mod flatten_layer;
pub mod max_pooling_layer;

pub enum LayerType {
    Convolutional(convolutional_layer::ConvolutionalLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
    Flatten(flatten_layer::FlattenLayer),
    Dense(dense_layer::DenseLayer),
}

// TODO: make it generic
// use ndarray::Array3;
// pub trait LayerTrait {
//     fn forward_propagate(&mut self, input: &Array3<f32>, is_training: bool) -> Array3<f32>;
//     fn backward_propagate(&mut self, error: &Array3<f32>) -> Array3<f32>;
// }
