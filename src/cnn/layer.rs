pub mod convolutional_layer;
pub mod flatten_layer;
pub mod max_pooling_layer;

pub enum LayerType {
    Convolutional(convolutional_layer::ConvolutionalLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
    Flatten(flatten_layer::FlattenLayer),
}
