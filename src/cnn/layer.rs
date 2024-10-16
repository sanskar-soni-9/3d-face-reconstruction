pub mod convolutional_layer;
pub mod dense_layer;
pub mod flatten_layer;
pub mod global_avg_pooling_layer;
pub mod max_pooling_layer;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Convolutional(convolutional_layer::ConvolutionalLayer),
    Dense(dense_layer::DenseLayer),
    Flatten(flatten_layer::FlattenLayer),
    GlobalAvgPooling(global_avg_pooling_layer::GlobalAvgPoolingLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
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
