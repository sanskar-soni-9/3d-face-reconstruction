pub mod activation_layer;
pub mod convolutional_layer;
pub mod dense_layer;
pub mod depthwise_conv_layer;
pub mod flatten_layer;
pub mod global_avg_pooling_layer;
pub mod max_pooling_layer;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Activation(activation_layer::ActivationLayer),
    Convolutional(convolutional_layer::ConvolutionalLayer),
    Dense(dense_layer::DenseLayer),
    DepthwiseConvLayer(depthwise_conv_layer::DepthwiseConvolutionalLayer),
    Flatten(flatten_layer::FlattenLayer),
    GlobalAvgPooling(global_avg_pooling_layer::GlobalAvgPoolingLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
}
