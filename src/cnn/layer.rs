mod activation_layer;
mod batch_norm_layer;
mod convolutional_layer;
mod dense_layer;
mod depthwise_conv_layer;
mod dropout_layer;
mod flatten_layer;
mod global_avg_pooling_layer;
mod max_pooling_layer;
mod operation_layer;
mod reshape_layer;

pub use activation_layer::*;
pub use batch_norm_layer::*;
pub use convolutional_layer::*;
pub use dense_layer::*;
pub use depthwise_conv_layer::*;
pub use dropout_layer::*;
pub use flatten_layer::*;
pub use global_avg_pooling_layer::*;
pub use max_pooling_layer::*;
pub use operation_layer::*;
pub use reshape_layer::*;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Activation(ActivationLayer),
    BatchNorm(BatchNormLayer),
    Convolutional(ConvolutionalLayer),
    Dense(DenseLayer),
    DepthwiseConv(DepthwiseConvolutionalLayer),
    Dropout(DropoutLayer),
    Flatten(FlattenLayer),
    GlobalAvgPooling(GlobalAvgPoolingLayer),
    MaxPooling(MaxPoolingLayer),
    Operand(OperandLayer),
    Operation(OperationLayer),
    Reshape(ReshapeLayer),
}
