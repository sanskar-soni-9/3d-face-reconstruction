use crate::config::{
    BATCH_EPSILON, DEFAULT_LEARNING_RATE, EPOCH_MODEL, INPUT_SHAPE, MODELS_DIR, NORM_MOMENTUM,
    TRAINIG_LABELS,
};
use crate::dataset::Labels;
use activation::Activation;
use cache::CNNCache;
use layer::*;
use ndarray::{s, Array1, Array2, Array3, Array4, Dim};
use rand::seq::SliceRandom;
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};

pub mod activation;
mod cache;
mod layer;

enum Tensor {
    Tensor4d(Array4<f64>),
    Tensor2d(Array2<f64>),
}

#[derive(Serialize, Deserialize)]
pub struct CNN {
    mini_batch_size: usize,
    layers: Vec<LayerType>,
    skip_layers: HashMap<usize, Vec<LayerType>>,
    epochs: usize,
    cur_epoch: usize,
    #[serde(skip)]
    data: Vec<Array3<f64>>,
    #[serde(skip)]
    lr: f64,
}

impl CNN {
    pub fn new(mini_batch_size: usize, epochs: usize, inputs: Vec<Array3<f64>>, lr: f64) -> Self {
        CNN {
            mini_batch_size,
            layers: vec![],
            skip_layers: HashMap::default(),
            epochs,
            cur_epoch: 0,
            data: inputs,
            lr,
        }
    }

    pub fn infer(&mut self, img_num: usize) -> Array1<f64> {
        let img_shape = self.data[img_num].shape();
        let mut input: Array4<f64> = Array4::zeros((1, img_shape[0], img_shape[1], img_shape[2]));
        input
            .slice_mut(s![0, .., .., ..])
            .assign(&self.data[img_num]);

        Self::forward_propagate(
            Tensor::Tensor4d(input),
            &mut self.layers,
            Some(&mut self.skip_layers),
            false,
        )
        .1
        .slice(s![0, ..])
        .to_owned()
    }

    /// Reference implementation
    /// - [keras-applications](https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py)
    fn add_resnet_v2_block(
        &mut self,
        mut input_shape: (usize, usize, usize, usize),
        filters: usize,
        kernel_size: usize,
        strides: usize,
        conv_shortcut: bool,
    ) -> (usize, usize, usize, usize) {
        let mut skip_id = self.layers.len() - 1;
        if conv_shortcut {
            skip_id += 2;
        } else {
            self.add_layer(LayerType::Operand(OperandLayer::new(
                skip_id,
                vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
            )));
        }

        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::ReLU);

        if conv_shortcut {
            self.add_layer(LayerType::Operand(OperandLayer::new(
                skip_id,
                vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
            )));
            let layer =
                ConvolutionalLayer::new(filters * 4, 1, strides, input_shape, Some(0.), false);
            self.add_skip_layer(skip_id, vec![LayerType::Convolutional(layer)]);
        } else if strides > 1 {
            let layer = MaxPoolingLayer::new(1, input_shape, strides, true);
            self.add_skip_layer(skip_id, vec![LayerType::MaxPooling(layer)]);
        }

        let layer = ConvolutionalLayer::new(filters, 1, 1, input_shape, None, false);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::ReLU);

        let layer = ConvolutionalLayer::new(filters, kernel_size, strides, input_shape, None, true);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::ReLU);

        let layer = ConvolutionalLayer::new(filters * 4, 1, 1, input_shape, Some(0.), false);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));

        self.add_layer(LayerType::Operation(OperationLayer::new(
            skip_id,
            vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
            OperationType::Add,
        )));
        input_shape
    }

    pub fn stack_resnet_v2_blocks(&mut self, filters: usize, strides: usize, blocks: usize) {
        let error_msg = "Add resnet block after a convolutional or max pool layer.";
        let mut input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => (
                self.mini_batch_size,
                INPUT_SHAPE.0,
                INPUT_SHAPE.1,
                INPUT_SHAPE.2,
            ),
        };
        input_shape = self.add_resnet_v2_block(input_shape, filters, 3, 1, true);
        for _ in 2..blocks {
            input_shape = self.add_resnet_v2_block(input_shape, filters, 3, 1, false);
        }
        self.add_resnet_v2_block(input_shape, filters, 3, strides, false);
    }

    pub fn add_activation_layer(&mut self, activation: Activation) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => layer.input_shape().to_owned(),
                LayerType::BatchNorm(layer) => layer.input_shape().to_owned(),
                LayerType::Convolutional(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dense(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::DepthwiseConv(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dropout(layer) => layer.input_shape().to_owned(),
                LayerType::Flatten(layer) => vec![layer.output_size()],
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Operand(layer) => layer.input_shape().to_owned(),
                LayerType::Operation(layer) => layer.input_shape().to_owned(),
                LayerType::Reshape(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
            },
            None => panic!("Activation Layer can not be the first layer."),
        };
        self.add_layer(LayerType::Activation(ActivationLayer::new(
            activation,
            input_shape,
        )));
    }

    pub fn add_batch_norm_layer(&mut self, axis: usize, epsilon: f64, momentum: f64) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => layer.input_shape().to_owned(),
                LayerType::BatchNorm(layer) => layer.input_shape().to_owned(),
                LayerType::Convolutional(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dense(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::DepthwiseConv(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dropout(layer) => layer.input_shape().to_owned(),
                LayerType::Flatten(layer) => vec![layer.output_size()],
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Operand(layer) => layer.input_shape().to_owned(),
                LayerType::Operation(layer) => layer.input_shape().to_owned(),
                LayerType::Reshape(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
            },
            None => panic!("Batch Layer can not be the first layer."),
        };
        self.add_layer(LayerType::BatchNorm(BatchNormLayer::new(
            axis,
            epsilon,
            input_shape,
            momentum,
        )));
    }

    pub fn add_convolutional_layer(
        &mut self,
        filters: usize,
        kernel_size: usize,
        strides: usize,
        bias: Option<f64>,
        add_padding: bool,
    ) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let error_msg = "Add convolutional layer after a convolutional or max pooling layer.";
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => (
                self.mini_batch_size,
                INPUT_SHAPE.0,
                INPUT_SHAPE.1,
                INPUT_SHAPE.2,
            ),
        };

        self.add_layer(LayerType::Convolutional(ConvolutionalLayer::new(
            filters,
            kernel_size,
            strides,
            input_shape,
            bias,
            add_padding,
        )));
    }

    pub fn add_mbconv_layer(
        &mut self,
        factor: usize,
        filters: usize,
        kernel_size: usize,
        strides: usize,
        se_ratio: f64,
        dropout_rate: f64,
        add_padding: bool,
    ) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let error_msg = "Add mbconv layer after a convolutional or max pool layer.";
        let mut input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => (
                self.mini_batch_size,
                INPUT_SHAPE.0,
                INPUT_SHAPE.1,
                INPUT_SHAPE.2,
            ),
        };

        let input_filters = input_shape.1;
        let addition_id = self.layers.len() - 1;
        let use_identity_skip = strides == 1 && filters == input_shape.1;

        if use_identity_skip {
            self.add_layer(LayerType::Operand(OperandLayer::new(
                addition_id,
                vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
            )));
        }

        // Expansion
        if factor != 1 {
            let layer = ConvolutionalLayer::new(
                input_filters * factor,
                1,
                1,
                input_shape,
                None,
                add_padding,
            );
            input_shape = layer.output_shape();
            self.add_layer(LayerType::Convolutional(layer));
            self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
            self.add_activation_layer(Activation::SiLU);
        }

        // Depthwise Convolution
        let layer =
            DepthwiseConvolutionalLayer::new(kernel_size, strides, input_shape, None, add_padding);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::DepthwiseConv(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::SiLU);

        // Squeeze and Excitation
        if !se_ratio.is_zero() {
            let reduced_filters = (input_filters as f64 * se_ratio).max(1.) as usize;
            let mul_id = self.layers.len() - 1;
            self.add_layer(LayerType::Operand(OperandLayer::new(
                mul_id,
                vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
            )));

            let layer = GlobalAvgPoolingLayer::new(input_shape);
            let reshape_input_shape = layer.output_shape();
            self.add_layer(LayerType::GlobalAvgPooling(layer));

            let target_shape = (reshape_input_shape.0, reshape_input_shape.1, 1, 1);
            let layer = ReshapeLayer::new(reshape_input_shape, target_shape, false);
            let mut se_input_shape = layer.output_shape();
            self.add_layer(LayerType::Reshape(layer));

            let layer =
                ConvolutionalLayer::new(reduced_filters, 1, 1, se_input_shape, Some(0.), true);
            se_input_shape = layer.output_shape();
            self.add_layer(LayerType::Convolutional(layer));
            self.add_activation_layer(Activation::SiLU);

            let layer = ConvolutionalLayer::new(
                input_filters * factor,
                1,
                1,
                se_input_shape,
                Some(0.),
                true,
            );
            se_input_shape = layer.output_shape();
            self.add_layer(LayerType::Convolutional(layer));
            self.add_activation_layer(Activation::Sigmoid);

            self.add_layer(LayerType::Operation(OperationLayer::new(
                mul_id,
                vec![
                    se_input_shape.0,
                    se_input_shape.1,
                    input_shape.2,
                    input_shape.3,
                ],
                OperationType::Mul,
            )));
        }

        // Output
        let layer = ConvolutionalLayer::new(filters, 1, 1, input_shape, None, add_padding);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);

        if use_identity_skip {
            if dropout_rate > 0. {
                self.add_layer(LayerType::Dropout(DropoutLayer::new(
                    vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
                    dropout_rate,
                )));
            }
            self.add_layer(LayerType::Operation(OperationLayer::new(
                addition_id,
                vec![input_shape.0, input_shape.1, input_shape.2, input_shape.3],
                OperationType::Add,
            )));
        }
    }

    pub fn add_dropout_layer(&mut self, dropout_rate: f64) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => layer.input_shape().to_owned(),
                LayerType::BatchNorm(layer) => layer.input_shape().to_owned(),
                LayerType::Convolutional(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dense(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::DepthwiseConv(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dropout(layer) => layer.input_shape().to_owned(),
                LayerType::Flatten(layer) => vec![layer.output_size()],
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Operand(layer) => layer.input_shape().to_owned(),
                LayerType::Operation(layer) => layer.input_shape().to_owned(),
                LayerType::Reshape(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
            },
            None => panic!("Batch Layer can not be the first layer."),
        };
        self.add_layer(LayerType::Dropout(DropoutLayer::new(
            input_shape,
            dropout_rate,
        )));
    }

    pub fn add_global_avg_pooling_layer(&mut self) {
        let error_msg = "Add global average pooling layer after a convolutional or pooling layer.";
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::DepthwiseConv(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => {
                panic!("{}", error_msg)
            }
        };
        self.add_layer(LayerType::GlobalAvgPooling(GlobalAvgPoolingLayer::new(
            input_shape,
        )));
    }

    pub fn add_max_pooling_layer(&mut self, kernel_size: usize, strides: usize, add_padding: bool) {
        let error_msg = "Add max pooling layer after a convolutional or max pooling layer.";
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let output_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => panic!("{}", error_msg),
        };

        self.add_layer(LayerType::MaxPooling(MaxPoolingLayer::new(
            kernel_size,
            output_shape,
            strides,
            add_padding,
        )));
    }

    pub fn add_flatten_layer(&mut self) {
        let error_msg = "Add flatten layer after a convolutional or max pooling layer.";
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::MaxPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1], shape[2], shape[3])
                }
                LayerType::Reshape(layer) => layer.output_shape(),
                _ => panic!("{}", error_msg),
            },
            None => panic!("{}", error_msg),
        };
        self.add_layer(LayerType::Flatten(FlattenLayer::new(input_shape)));
    }

    pub fn add_dense_layer(&mut self, neurons: usize, bias: f64) {
        let error_msg = "Add dense layer after a flatten, global average or dense layer.";
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 2 {
                        (shape[0], shape[1])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::BatchNorm(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 2 {
                        (shape[0], shape[1])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Dense(layer) => layer.output_shape(),
                LayerType::Dropout(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 2 {
                        (shape[0], shape[1])
                    } else {
                        panic!("{}", error_msg);
                    }
                }
                LayerType::Flatten(layer) => (
                    self.mini_batch_size,
                    layer.input_shape().0 * layer.input_shape().1 * layer.input_shape().2,
                ),
                LayerType::GlobalAvgPooling(layer) => layer.output_shape(),
                LayerType::Operand(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1])
                }
                LayerType::Operation(layer) => {
                    let shape = layer.input_shape();
                    (shape[0], shape[1])
                }
                _ => panic!("{}", error_msg),
            },
            None => panic!("{}", error_msg),
        };

        self.add_layer(LayerType::Dense(DenseLayer::new(
            input_shape,
            neurons,
            bias,
        )));
    }

    pub fn train(&mut self, labels: Vec<Labels>) {
        for e in self.cur_epoch..self.epochs {
            let mut rng = rand::thread_rng();
            let mut shuffled_batch: Vec<_> = (0..self.data.len()).collect();
            shuffled_batch.shuffle(&mut rng);
            for batch in shuffled_batch.chunks(self.mini_batch_size) {
                let mut input: Array4<f64> = Array4::zeros((
                    self.mini_batch_size,
                    INPUT_SHAPE.0,
                    INPUT_SHAPE.1,
                    INPUT_SHAPE.2,
                ));
                let mut training_labels: Vec<Array1<f64>> =
                    Vec::with_capacity(self.mini_batch_size);

                for (batch_i, data_i) in batch.iter().enumerate() {
                    input
                        .slice_mut(s![batch_i, .., .., ..])
                        .assign(&self.data[*data_i]);
                    training_labels.push(self.prepare_training_labels(&labels[*data_i]));
                }
                for pad_i in batch.len()..self.mini_batch_size {
                    let data_i = shuffled_batch.choose(&mut rng).unwrap();
                    input
                        .slice_mut(s![pad_i, .., .., ..])
                        .assign(&self.data[*data_i]);
                    training_labels.push(self.prepare_training_labels(&labels[*data_i]));
                }

                let start_time = std::time::SystemTime::now();
                let (_, prediction, mut store) = Self::forward_propagate(
                    Tensor::Tensor4d(input),
                    &mut self.layers,
                    Some(&mut self.skip_layers),
                    true,
                );
                let forward_time = std::time::SystemTime::now();
                println!(
                    "\nFORWARD TOOK: {:?}\n",
                    forward_time.duration_since(start_time)
                );

                let mut error: Array2<f64> = Array2::zeros(prediction.raw_dim());
                error
                    .outer_iter_mut()
                    .enumerate()
                    .for_each(|(idx, mut err)| {
                        err.assign(&(&prediction.slice(s![idx, ..]) - &training_labels[idx]));
                    });
                println!("PREDICTION: {:?}\nERROR: {:?}\n", prediction, error);

                Self::backward_propagate(
                    Tensor::Tensor2d(error),
                    &mut self.layers,
                    Some(&mut self.skip_layers),
                    Some(&mut store),
                    self.lr,
                );
                let backward_time = std::time::SystemTime::now();
                println!(
                    "BACKWARD TOOK: {:?}\nIMAGE TOOK: {:?}\n",
                    backward_time.duration_since(forward_time),
                    backward_time.duration_since(start_time)
                );
            }
            self.cur_epoch = e + 1;
            println!("Epoch {} complete saving model.\n", self.cur_epoch);
            self.save(&format!("{}{}", EPOCH_MODEL, self.cur_epoch));
        }
    }

    fn forward_propagate(
        input: Tensor,
        layers: &mut [LayerType],
        mut skip_layers: Option<&mut HashMap<usize, Vec<LayerType>>>,
        is_training: bool,
    ) -> (Array4<f64>, Array2<f64>, CNNCache) {
        let cache_err = "No cache found for backpropagation calculations.";
        let mut store = CNNCache::default();

        let (mut input, mut flatten_input) = match input {
            Tensor::Tensor4d(input) => (input, Array2::default((0, 0))),
            Tensor::Tensor2d(flatten_input) => (Array4::default((0, 0, 0, 0)), flatten_input),
        };

        for layer in layers.iter_mut() {
            match layer {
                LayerType::Activation(activation_layer) => {
                    if activation_layer.input_shape().len() == 2 {
                        flatten_input =
                            activation_layer.forward_propagate(&flatten_input, is_training);
                    } else {
                        input = activation_layer.forward_propagate(&input, is_training);
                    }
                }
                LayerType::BatchNorm(batchnorm_layer) => {
                    if batchnorm_layer.input_shape().len() == 2 {
                        let cache: Option<BNCache<Dim<[usize; 2]>>>;
                        (flatten_input, cache) =
                            batchnorm_layer.forward_propagate(flatten_input, is_training);
                        if let Some(cache) = cache {
                            store.add_bn2(cache);
                        }
                    } else {
                        let cache: Option<BNCache<Dim<[usize; 4]>>>;
                        (input, cache) = batchnorm_layer.forward_propagate(input, is_training);
                        if let Some(cache) = cache {
                            store.add_bn4(cache);
                        }
                    }
                }
                LayerType::Convolutional(convolutional_layer) => {
                    input = convolutional_layer.forward_propagate(input, is_training);
                }
                LayerType::Dense(dense_layer) => {
                    flatten_input = dense_layer.forward_propagate(flatten_input, is_training);
                }
                LayerType::DepthwiseConv(depthwise_conv_layer) => {
                    input = depthwise_conv_layer.forward_propagate(input, is_training);
                }
                LayerType::Dropout(dropout_layer) => {
                    if dropout_layer.input_shape().len() == 2 {
                        flatten_input = dropout_layer.forward_propagate(flatten_input, is_training);
                    } else {
                        input = dropout_layer.forward_propagate(input, is_training);
                    }
                }
                LayerType::Flatten(flatten_layer) => {
                    flatten_input = flatten_layer.forward_propagate(&input, is_training);
                }
                LayerType::GlobalAvgPooling(avg_pooling_layer) => {
                    flatten_input = avg_pooling_layer.forward_propagate(&input, is_training);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    input = max_pooling_layer.forward_propagate(input, is_training);
                }
                LayerType::Operand(operand_layer) => {
                    let id = operand_layer.id();
                    if operand_layer.input_shape().len() == 2 {
                        let mut cache: OperandCache<Dim<[usize; 2]>>;
                        (flatten_input, cache) =
                            operand_layer.forward_propagate(flatten_input, is_training);
                        if let Some(ref mut skip_layers) = skip_layers {
                            if let Some(skip_layers) = skip_layers.get_mut(&id) {
                                let (_, skip_actvns, _) = Self::forward_propagate(
                                    Tensor::Tensor2d(flatten_input.clone()),
                                    skip_layers,
                                    None,
                                    is_training,
                                );
                                cache.update_skip(skip_actvns);
                            }
                        }
                        store.add_operand2d(id, cache);
                    } else {
                        let mut cache: OperandCache<Dim<[usize; 4]>>;
                        (input, cache) = operand_layer.forward_propagate(input, is_training);
                        if let Some(ref mut skip_layers) = skip_layers {
                            if let Some(skip_layers) = skip_layers.get_mut(&id) {
                                let (skip_actvns, _, _) = Self::forward_propagate(
                                    Tensor::Tensor4d(input.clone()),
                                    skip_layers,
                                    None,
                                    is_training,
                                );
                                cache.update_skip(skip_actvns);
                            }
                        }
                        store.add_operand4d(id, cache);
                    }
                }
                LayerType::Operation(operation_layer) => {
                    let id = operation_layer.operand_id();
                    if operation_layer.input_shape().len() == 2 {
                        let (id, mut cache) = store.consume_operand2d(id).expect(cache_err);
                        (flatten_input, cache) =
                            operation_layer.forward_propagate(flatten_input, cache, is_training);
                        store.add_operand2d(id, cache);
                    } else {
                        let (id, mut cache) = store.consume_operand4d(id).expect(cache_err);
                        (input, cache) =
                            operation_layer.forward_propagate(input, cache, is_training);
                        store.add_operand4d(id, cache);
                    }
                }
                LayerType::Reshape(reshape_layer) => {
                    input = reshape_layer.forward_propagate(&flatten_input, is_training);
                }
            };
        }
        (input, flatten_input, store)
    }

    fn backward_propagate(
        error: Tensor,
        layers: &mut [LayerType],
        mut skip_layers: Option<&mut HashMap<usize, Vec<LayerType>>>,
        mut store: Option<&mut CNNCache>,
        lr: f64,
    ) -> (Array2<f64>, Array4<f64>) {
        let cache_err = "No cache found for backpropagation calculations.";
        let (mut shaped_error, mut error) = match error {
            Tensor::Tensor4d(shaped_error) => (shaped_error, Array2::default((0, 0))),
            Tensor::Tensor2d(error) => (Array4::default((0, 0, 0, 0)), error),
        };

        for layer in layers.iter_mut().rev() {
            match layer {
                LayerType::Activation(activation_layer) => {
                    if activation_layer.input_shape().len() == 2 {
                        error = activation_layer.backward_propagate(error, lr);
                    } else {
                        shaped_error = activation_layer.backward_propagate(shaped_error, lr);
                    }
                }
                LayerType::BatchNorm(batchnorm_layer) => {
                    if batchnorm_layer.input_shape().len() == 2 {
                        let cache = store.as_mut().unwrap().consume_bn2().expect(cache_err);
                        error = batchnorm_layer.backward_propagate(error, cache, lr);
                    } else {
                        let cache = store.as_mut().unwrap().consume_bn4().expect(cache_err);
                        shaped_error = batchnorm_layer.backward_propagate(shaped_error, cache, lr);
                    }
                }
                LayerType::Convolutional(convolutional_layer) => {
                    shaped_error = convolutional_layer.backward_propagate(shaped_error, lr);
                }
                LayerType::Dense(dense_layer) => {
                    error = dense_layer.backward_propagate(error, lr);
                }
                LayerType::DepthwiseConv(depthwise_conv_layer) => {
                    shaped_error = depthwise_conv_layer.backward_propagate(shaped_error, lr);
                }
                LayerType::Dropout(dropout_layer) => {
                    if dropout_layer.input_shape().len() == 2 {
                        error = dropout_layer.backward_propagate(error, lr);
                    } else {
                        shaped_error = dropout_layer.backward_propagate(shaped_error, lr);
                    }
                }
                LayerType::Flatten(flatten_layer) => {
                    shaped_error = flatten_layer.backward_propagate(&error);
                }
                LayerType::GlobalAvgPooling(avg_pooling_layer) => {
                    shaped_error = avg_pooling_layer.backward_propagate(&error);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    shaped_error = max_pooling_layer.backward_propagate(shaped_error);
                }
                LayerType::Operand(operand_layer) => {
                    let operand_id = operand_layer.id();
                    if operand_layer.input_shape().len() == 2 {
                        let (_, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand2d(operand_id)
                            .expect(cache_err);
                        if let Some(ref mut skip_layers) = skip_layers {
                            if let Some(skip_layers) = skip_layers.get_mut(&operand_id) {
                                (cache.skip_actvns, _) = Self::backward_propagate(
                                    Tensor::Tensor2d(cache.skip_actvns),
                                    skip_layers,
                                    None,
                                    None,
                                    lr,
                                );
                            }
                        }
                        error = operand_layer.backward_propagate(error, cache, lr);
                    } else {
                        let (_, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand4d(operand_id)
                            .expect(cache_err);
                        if let Some(ref mut skip_layers) = skip_layers {
                            if let Some(skip_layers) = skip_layers.get_mut(&operand_id) {
                                (_, cache.skip_actvns) = Self::backward_propagate(
                                    Tensor::Tensor4d(cache.skip_actvns),
                                    skip_layers,
                                    None,
                                    None,
                                    lr,
                                );
                            }
                        }
                        shaped_error = operand_layer.backward_propagate(shaped_error, cache, lr);
                    }
                }
                LayerType::Operation(operation_layer) => {
                    let operand_id = operation_layer.operand_id();
                    if operation_layer.input_shape().len() == 2 {
                        let (operand_id, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand2d(operand_id)
                            .expect(cache_err);
                        (error, cache) = operation_layer.backward_propagate(error, cache, lr);
                        store.as_mut().unwrap().add_operand2d(operand_id, cache);
                    } else {
                        let (operand_id, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand4d(operand_id)
                            .expect(cache_err);
                        (shaped_error, cache) =
                            operation_layer.backward_propagate(shaped_error, cache, lr);
                        store.as_mut().unwrap().add_operand4d(operand_id, cache);
                    }
                }
                LayerType::Reshape(reshape_layer) => {
                    error = reshape_layer.backward_propagate(&shaped_error, lr);
                }
            }
        }
        (error, shaped_error)
    }

    fn prepare_training_labels(&self, labels: &Labels) -> Array1<f64> {
        let mut training_labels: Vec<f64> = vec![];
        for train_label in TRAINIG_LABELS {
            match train_label {
                "pts_2d" => training_labels.append(&mut labels.pts_2d().to_vec()),
                "pts_3d" => training_labels.append(&mut labels.pts_3d().to_vec()),
                "pose_para" => training_labels.append(&mut labels.pose_para().to_vec()),
                "shape_para" => training_labels.append(&mut labels.shape_para().to_vec()),
                "illum_para" => training_labels.append(&mut labels.illum_para().to_vec()),
                "color_para" => training_labels.append(&mut labels.color_para().to_vec()),
                "exp_para" => training_labels.append(&mut labels.exp_para().to_vec()),
                "tex_para" => training_labels.append(&mut labels.tex_para().to_vec()),
                "pt2d" => training_labels.append(&mut labels.pts_2d().to_vec()),
                "roi" => training_labels.append(&mut labels.roi().to_vec()),
                _ => panic!("Unknown Label: {:?}", train_label),
            }
        }
        Array1::from_vec(training_labels)
    }

    fn add_layer(&mut self, layer: LayerType) {
        self.layers.push(layer);
    }

    fn add_skip_layer(&mut self, skip_id: usize, layers: Vec<LayerType>) {
        self.skip_layers.insert(skip_id, layers);
    }

    fn save(&self, file_name: &str) {
        let mut model_file = std::fs::File::create(format!("{}/{}", MODELS_DIR, file_name,))
            .unwrap_or_else(|e| panic!("Error creating model file: {}\nError: {}", file_name, e));
        let json = serde_json::to_string(&self)
            .unwrap_or_else(|e| panic!("Error serializing model: {}\nError: {}", file_name, e));
        model_file
            .write_all(json.as_bytes())
            .unwrap_or_else(|e| panic!("Error writing to model file: {}\nError: {}", file_name, e));
    }

    pub fn load(model_path: &str) -> CNN {
        let mut file = std::fs::File::open(model_path)
            .unwrap_or_else(|e| panic!("Error opening model file: {}\nError: {}", model_path, e));
        let mut model = String::new();
        file.read_to_string(&mut model)
            .unwrap_or_else(|e| panic!("Error reading model file: {}\nError: {}", model_path, e));
        serde_json::from_str(&model)
            .unwrap_or_else(|e| panic!("Error deserializing model: {}\nError: {}", model_path, e))
    }

    pub fn load_with_data(model_path: &str, data: Vec<Array3<f64>>, lr: Option<&str>) -> CNN {
        let mut cnn = Self::load(model_path);
        cnn.data = data;
        cnn.lr = match lr {
            Some(lr) => lr.parse().expect("Learning rate should be a valid f64"),
            None => DEFAULT_LEARNING_RATE,
        };
        cnn
    }
}
