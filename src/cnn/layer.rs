pub mod convolutional_layer;
pub mod dense_layer;
pub mod flatten_layer;
pub mod max_pooling_layer;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Convolutional(convolutional_layer::ConvolutionalLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
    Flatten(flatten_layer::FlattenLayer),
    Dense(dense_layer::DenseLayer),
}

pub fn relu(x: f64) -> f64 {
    if x > 0.0 {
        return x;
    }
    0.0
}

pub fn relu_prime(x: f64) -> f64 {
    if x > 0.0 {
        return 1.0;
    }
    0.0
}
