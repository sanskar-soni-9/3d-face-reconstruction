pub mod activation_layer;
pub mod batch_norm_layer;
pub mod convolutional_layer;
pub mod dense_layer;
pub mod depthwise_conv_layer;
pub mod dropout_layer;
pub mod flatten_layer;
pub mod global_avg_pooling_layer;
pub mod max_pooling_layer;
pub mod operation_layer;
pub mod reshape_layer;

#[derive(serde::Serialize, serde::Deserialize)]
pub enum LayerType {
    Activation(activation_layer::ActivationLayer),
    BatchNorm(batch_norm_layer::BatchNormLayer),
    Convolutional(convolutional_layer::ConvolutionalLayer),
    Dense(dense_layer::DenseLayer),
    DepthwiseConv(depthwise_conv_layer::DepthwiseConvolutionalLayer),
    Dropout(dropout_layer::DropoutLayer),
    Flatten(flatten_layer::FlattenLayer),
    GlobalAvgPooling(global_avg_pooling_layer::GlobalAvgPoolingLayer),
    MaxPooling(max_pooling_layer::MaxPoolingLayer),
    Operand(operation_layer::OperandLayer),
    Operation(operation_layer::OperationLayer),
    Reshape(reshape_layer::ReshapeLayer),
}
