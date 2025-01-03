use crate::{
    config::{BATCH_EPSILON, EPOCH_MODEL, INPUT_SHAPE, MODELS_DIR, NORM_MOMENTUM, PRETTY_SAVE},
    data_loader::DataLoader,
    dataset::Labels,
};
use activation::Activation;
use cache::CNNCache;
use layer::*;
use ndarray::{Array2, Array4, Axis, Dim};
use rand_distr::num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    io::{Read, Write},
};

pub mod activation;
mod cache;
mod layer;
pub mod optimizer;

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
    lr: f64,
    optimizer: optimizer::OptimizerType,
    training_data_percent: f64,
}

impl CNN {
    pub fn new(
        lr: f64,
        mini_batch_size: usize,
        epochs: usize,
        training_data_percent: Option<f64>,
        optimizer: optimizer::OptimizerType,
    ) -> Self {
        CNN {
            lr,
            mini_batch_size,
            layers: vec![],
            skip_layers: HashMap::default(),
            epochs,
            cur_epoch: 0,
            optimizer,
            training_data_percent: training_data_percent.unwrap_or(0.8),
        }
    }

    fn compute_input_shape(&self) -> Vec<usize> {
        match self.layers.last() {
            Some(last_layer) => match last_layer {
                LayerType::Activation(layer) => layer.output_shape().to_vec(),
                LayerType::BatchNorm(layer) => layer.output_shape().to_vec(),
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
                LayerType::Dropout(layer) => layer.output_shape().to_vec(),
                LayerType::Flatten(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Operand(layer) => layer.output_shape().to_vec(),
                LayerType::Operation(layer) => layer.output_shape().to_vec(),
                LayerType::Reshape(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
            },
            None => {
                vec![
                    self.mini_batch_size,
                    INPUT_SHAPE.0,
                    INPUT_SHAPE.1,
                    INPUT_SHAPE.2,
                ]
            }
        }
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
            let layer = ConvolutionalLayer::new(
                filters * 4,
                1,
                strides,
                input_shape,
                Some(0.),
                false,
                &self.optimizer,
            );
            self.add_skip_layer(skip_id, vec![LayerType::Convolutional(layer)]);
        } else if strides > 1 {
            let layer = MaxPoolingLayer::new(1, input_shape, strides, true);
            self.add_skip_layer(skip_id, vec![LayerType::MaxPooling(layer)]);
        }

        let layer =
            ConvolutionalLayer::new(filters, 1, 1, input_shape, None, false, &self.optimizer);
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::ReLU);

        let layer = ConvolutionalLayer::new(
            filters,
            kernel_size,
            strides,
            input_shape,
            None,
            true,
            &self.optimizer,
        );
        input_shape = layer.output_shape();
        self.add_layer(LayerType::Convolutional(layer));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::ReLU);

        let layer = ConvolutionalLayer::new(
            filters * 4,
            1,
            1,
            input_shape,
            Some(0.),
            false,
            &self.optimizer,
        );
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
        let input_shape = self.compute_input_shape();
        let mut input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add resnet block after a convolutional or max pool layer.");
        };
        input_shape = self.add_resnet_v2_block(input_shape, filters, 3, 1, true);
        for _ in 2..blocks {
            input_shape = self.add_resnet_v2_block(input_shape, filters, 3, 1, false);
        }
        self.add_resnet_v2_block(input_shape, filters, 3, strides, false);
    }

    pub fn add_activation_layer(&mut self, activation: Activation) {
        let input_shape = self.compute_input_shape();
        self.add_layer(LayerType::Activation(ActivationLayer::new(
            activation,
            input_shape,
        )));
    }

    pub fn add_batch_norm_layer(&mut self, axis: usize, epsilon: f64, momentum: f64) {
        let input_shape = self.compute_input_shape();
        self.add_layer(LayerType::BatchNorm(BatchNormLayer::new(
            axis,
            epsilon,
            input_shape,
            momentum,
            &self.optimizer,
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
        let input_shape = self.compute_input_shape();
        let input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add convolutional layer after a convolutional or max pooling layer.");
        };

        self.add_layer(LayerType::Convolutional(ConvolutionalLayer::new(
            filters,
            kernel_size,
            strides,
            input_shape,
            bias,
            add_padding,
            &self.optimizer,
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
        let input_shape = self.compute_input_shape();
        let mut input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add mbconv layer after a convolutional or max pool layer.");
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
                &self.optimizer,
            );
            input_shape = layer.output_shape();
            self.add_layer(LayerType::Convolutional(layer));
            self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
            self.add_activation_layer(Activation::SiLU);
        }

        // Depthwise Convolution
        let layer = DepthwiseConvolutionalLayer::new(
            kernel_size,
            strides,
            input_shape,
            None,
            add_padding,
            &self.optimizer,
        );
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

            let layer = ConvolutionalLayer::new(
                reduced_filters,
                1,
                1,
                se_input_shape,
                Some(0.),
                true,
                &self.optimizer,
            );
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
                &self.optimizer,
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
        let layer = ConvolutionalLayer::new(
            filters,
            1,
            1,
            input_shape,
            None,
            add_padding,
            &self.optimizer,
        );
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
        let input_shape = self.compute_input_shape();
        self.add_layer(LayerType::Dropout(DropoutLayer::new(
            input_shape,
            dropout_rate,
        )));
    }

    pub fn add_global_avg_pooling_layer(&mut self) {
        let input_shape = self.compute_input_shape();
        let input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add global average pooling layer after a convolutional or pooling layer.");
        };
        self.add_layer(LayerType::GlobalAvgPooling(GlobalAvgPoolingLayer::new(
            input_shape,
        )));
    }

    pub fn add_max_pooling_layer(&mut self, kernel_size: usize, strides: usize, add_padding: bool) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let input_shape = self.compute_input_shape();
        let input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add max pooling layer after a convolutional or max pooling layer.");
        };

        self.add_layer(LayerType::MaxPooling(MaxPoolingLayer::new(
            kernel_size,
            input_shape,
            strides,
            add_padding,
        )));
    }

    pub fn add_flatten_layer(&mut self) {
        let input_shape = self.compute_input_shape();
        let input_shape = if input_shape.len() == 4 {
            (
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            )
        } else {
            panic!("Add flatten layer after a convolutional or max pooling layer.");
        };

        self.add_layer(LayerType::Flatten(FlattenLayer::new(input_shape)));
    }

    pub fn add_dense_layer(&mut self, neurons: usize, bias: f64) {
        let input_shape = self.compute_input_shape();
        let input_shape = if input_shape.len() == 2 {
            (input_shape[0], input_shape[1])
        } else {
            panic!("Add dense layer after a flatten, global average or dense layer.");
        };

        self.add_layer(LayerType::Dense(DenseLayer::new(
            input_shape,
            neurons,
            bias,
            &self.optimizer,
        )));
    }

    pub fn train(&mut self, mut labels: Vec<Labels>) {
        let train_labels_range =
            0..(labels.len() as f64 * self.training_data_percent).floor() as usize;
        let mut training_data_loader =
            DataLoader::new(self.mini_batch_size, true, &mut labels[train_labels_range]);

        for e in self.cur_epoch..self.epochs {
            for (input, training_labels) in training_data_loader.by_ref() {
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

                let error: Array2<f64> = &prediction - &training_labels;
                let mse = error.powi(2).mean_axis(Axis(1)).unwrap().mean_axis(Axis(0));
                let mse_grd = &error * 2. / error.shape().iter().product::<usize>() as f64;
                println!(
                    "PREDICTION: {:?}\nERROR: {:?}\nMSE: {:?}\n",
                    prediction, error, mse
                );

                Self::backward_propagate(
                    Tensor::Tensor2d(mse_grd),
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
            training_data_loader.reset_iter();
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
                    if activation_layer.output_shape().len() == 2 {
                        error = activation_layer.backward_propagate(error);
                    } else {
                        shaped_error = activation_layer.backward_propagate(shaped_error);
                    }
                }
                LayerType::BatchNorm(batchnorm_layer) => {
                    if batchnorm_layer.output_shape().len() == 2 {
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
                    if dropout_layer.output_shape().len() == 2 {
                        error = dropout_layer.backward_propagate(error);
                    } else {
                        shaped_error = dropout_layer.backward_propagate(shaped_error);
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
                    if operand_layer.output_shape().len() == 2 {
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
                        error = operand_layer.backward_propagate(error, cache);
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
                        shaped_error = operand_layer.backward_propagate(shaped_error, cache);
                    }
                }
                LayerType::Operation(operation_layer) => {
                    let operand_id = operation_layer.operand_id();
                    if operation_layer.output_shape().len() == 2 {
                        let (operand_id, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand2d(operand_id)
                            .expect(cache_err);
                        (error, cache) = operation_layer.backward_propagate(error, cache);
                        store.as_mut().unwrap().add_operand2d(operand_id, cache);
                    } else {
                        let (operand_id, mut cache) = store
                            .as_mut()
                            .unwrap()
                            .consume_operand4d(operand_id)
                            .expect(cache_err);
                        (shaped_error, cache) =
                            operation_layer.backward_propagate(shaped_error, cache);
                        store.as_mut().unwrap().add_operand4d(operand_id, cache);
                    }
                }
                LayerType::Reshape(reshape_layer) => {
                    error = reshape_layer.backward_propagate(&shaped_error);
                }
            }
        }
        (error, shaped_error)
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
        let json = if PRETTY_SAVE {
            serde_json::to_string_pretty(&self).unwrap_or_else(|e| {
                panic!(
                    "Error while pretty serializing model: {}\nError: {}",
                    file_name, e
                )
            })
        } else {
            serde_json::to_string(&self).unwrap_or_else(|e| {
                panic!("Error while serializing model: {}\nError: {}", file_name, e)
            })
        };
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
}
