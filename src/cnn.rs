use crate::config::{
    BATCH_EPSILON, DEFAULT_LEARNING_RATE, EPOCH_MODEL, INPUT_SHAPE, MODELS_DIR, NORM_MOMENTUM,
    TRAINIG_LABELS,
};
use crate::dataset::Labels;
use activation::Activation;
use cache::CNNCache;
use layer::{
    activation_layer::*, batch_norm_layer::*, convolutional_layer::*, dense_layer::*,
    depthwise_conv_layer::*, flatten_layer::*, global_avg_pooling_layer::*, max_pooling_layer::*,
    LayerType,
};
use ndarray::{s, Array1, Array2, Array3, Array4, Dim};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};

pub mod activation;
mod cache;
mod layer;

#[derive(Serialize, Deserialize)]
pub struct CNN {
    mini_batch_size: usize,
    layers: Vec<LayerType>,
    epochs: usize,
    cur_epoch: usize,
    #[serde(skip)]
    data: Vec<Array3<f64>>,
    #[serde(skip)]
    store: CNNCache,
    lr: f64,
}

impl CNN {
    pub fn new(mini_batch_size: usize, epochs: usize, inputs: Vec<Array3<f64>>, lr: f64) -> Self {
        CNN {
            mini_batch_size,
            layers: vec![],
            epochs,
            cur_epoch: 0,
            data: inputs,
            store: CNNCache::default(),
            lr,
        }
    }

    pub fn infer(&mut self, img_num: usize) -> Array1<f64> {
        let img_shape = self.data[img_num].shape();
        let mut input: Array4<f64> = Array4::zeros((1, img_shape[0], img_shape[1], img_shape[2]));
        input
            .slice_mut(s![0, .., .., ..])
            .assign(&self.data[img_num]);
        self.forward_propagate(input, false)
            .slice(s![0, ..])
            .to_owned()
    }

    pub fn add_activation_layer(&mut self, activation: Activation) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => layer.input_shape().to_owned(),
                LayerType::BatchNormLayer(layer) => layer.input_shape().to_owned(),
                LayerType::Convolutional(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dense(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::DepthwiseConvLayer(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Flatten(layer) => vec![layer.output_size()],
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
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
                LayerType::BatchNormLayer(layer) => layer.input_shape().to_owned(),
                LayerType::Convolutional(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Dense(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::DepthwiseConvLayer(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
                LayerType::Flatten(layer) => vec![layer.output_size()],
                LayerType::GlobalAvgPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1]
                }
                LayerType::MaxPooling(layer) => {
                    let shape = layer.output_shape();
                    vec![shape.0, shape.1, shape.2, shape.3]
                }
            },
            None => panic!("Batch Layer can not be the first layer."),
        };
        self.add_layer(LayerType::BatchNormLayer(BatchNormLayer::new(
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
        add_padding: bool,
    ) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!(
                            "Add convolutional layer after a convolutional or max pooling layer."
                        );
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!(
                            "Add convolutional layer after a convolutional or max pooling layer."
                        );
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::MaxPooling(layer) => layer.output_shape(),
                _ => panic!("Add convolutional layer after a convolutional or max pooling layer."),
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
            add_padding,
        )));
    }

    pub fn add_mbconv_layer(
        &mut self,
        factor: usize,
        filters: usize,
        kernel_size: usize,
        strides: usize,
        add_padding: bool,
    ) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add mbconv layer after a convolutional or max pool layer.");
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add mbconv layer after a convolutional or max pooling layer.");
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::MaxPooling(layer) => layer.output_shape(),
                _ => panic!("Add mbconv layer after a convolutional or max pool layer."),
            },
            None => (
                self.mini_batch_size,
                INPUT_SHAPE.0,
                INPUT_SHAPE.1,
                INPUT_SHAPE.2,
            ),
        };

        let layer1 =
            ConvolutionalLayer::new(input_shape.0 * factor, 1, 1, input_shape, add_padding);
        let layer2 = DepthwiseConvolutionalLayer::new(
            kernel_size,
            strides,
            layer1.output_shape(),
            add_padding,
        );
        let layer3 = ConvolutionalLayer::new(filters, 1, 1, layer2.output_shape(), add_padding);

        self.add_layer(LayerType::Convolutional(layer1));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::SiLU);
        self.add_layer(LayerType::DepthwiseConvLayer(layer2));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
        self.add_activation_layer(Activation::SiLU);
        self.add_layer(LayerType::Convolutional(layer3));
        self.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
    }

    pub fn add_global_avg_pooling_layer(&mut self) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!(
                            "Add global average pooling layer after a convolutional or pooling layer."
                        );
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add global average pooling layer after a convolutional or pooling layer.");
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::MaxPooling(layer) => layer.output_shape(),
                _ => panic!(
                    "Add global average pooling layer after a convolutional or pooling layer."
                ),
            },
            None => {
                panic!("Add global average pooling layer after a convolutional or pooling layer.")
            }
        };
        self.add_layer(LayerType::GlobalAvgPooling(GlobalAvgPoolingLayer::new(
            input_shape,
        )));
    }

    pub fn add_max_pooling_layer(&mut self, kernel_size: usize, strides: usize) {
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
                        panic!("Add max pooling layer after a convolutional or max pooling layer.");
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add max pooling layer after a convolutional or max pooling layer.");
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::MaxPooling(layer) => layer.output_shape(),
                _ => panic!("Add max pooling layer after a convolutional or max pooling layer."),
            },
            None => panic!("Add max pooling layer after a convolutional or max pooling layer."),
        };

        self.add_layer(LayerType::MaxPooling(MaxPoolingLayer::new(
            kernel_size,
            output_shape,
            strides,
        )));
    }

    pub fn add_flatten_layer(&mut self) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add flatten layer after a convolutional or max pooling layer.");
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 4 {
                        (shape[0], shape[1], shape[2], shape[3])
                    } else {
                        panic!("Add flatten layer after a convolutional or max pooling layer.");
                    }
                }
                LayerType::Convolutional(layer) => layer.output_shape(),
                LayerType::MaxPooling(layer) => layer.output_shape(),
                _ => panic!("Add flatten layer after a convolutional or max pooling layer."),
            },
            None => panic!("Add flatten layer after a convolutional or max pooling layer."),
        };
        self.add_layer(LayerType::Flatten(FlattenLayer::new(input_shape)));
    }

    pub fn add_dense_layer(&mut self, neurons: usize, bias: f64) {
        let input_shape = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Activation(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 2 {
                        (shape[0], shape[1])
                    } else {
                        panic!("Add dense layer after a flatten, global average or dense layer.");
                    }
                }
                LayerType::BatchNormLayer(layer) => {
                    let shape = layer.input_shape();
                    if shape.len() == 2 {
                        (shape[0], shape[1])
                    } else {
                        panic!("Add dense layer after a flatten, global average or dense layer.");
                    }
                }
                LayerType::Dense(layer) => layer.output_shape(),
                LayerType::Flatten(layer) => (
                    self.mini_batch_size,
                    layer.input_shape().0 * layer.input_shape().1 * layer.input_shape().2,
                ),
                LayerType::GlobalAvgPooling(layer) => layer.output_shape(),
                _ => panic!("Add dense layer after a flatten, global average or dense layer."),
            },
            None => panic!("Add dense layer after a flatten, global average or dense layer."),
        };

        self.add_layer(LayerType::Dense(DenseLayer::new(
            input_shape,
            neurons,
            bias,
        )));
    }

    pub fn train(&mut self, labels: Vec<Labels>) {
        for e in self.cur_epoch..self.epochs {
            for batch in (0..labels.len())
                .take(self.data.len())
                .step_by(self.mini_batch_size)
            {
                if batch + self.mini_batch_size >= self.data.len() {
                    break;
                }

                let mut input: Array4<f64> = Array4::zeros((
                    self.mini_batch_size,
                    INPUT_SHAPE.0,
                    INPUT_SHAPE.1,
                    INPUT_SHAPE.2,
                ));
                let mut training_labels: Vec<Array1<f64>> = vec![];
                for j in 0..self.mini_batch_size {
                    input
                        .slice_mut(s![j, .., .., ..])
                        .assign(&self.data[batch + j]);
                    training_labels.push(self.prepare_training_labels(&labels[batch + j]));
                }

                let start_time = std::time::SystemTime::now();
                let prediction = self.forward_propagate(input, true);
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

                self.backward_propagate(error);
                let backward_time = std::time::SystemTime::now();
                println!(
                    "BACKWARD TOOK: {:?}\n\nIMAGE TOOK: {:?}\n",
                    backward_time.duration_since(forward_time),
                    backward_time.duration_since(start_time)
                );
            }
            self.cur_epoch = e + 1;
            println!("Epoch {} complete saving model.\n", self.cur_epoch);
            self.save(&format!("{}{}", EPOCH_MODEL, self.cur_epoch));
        }
    }

    fn forward_propagate(&mut self, mut input: Array4<f64>, is_training: bool) -> Array2<f64> {
        self.store.clear();
        let mut flatten_input: Array2<f64> = Array2::zeros((0, 0));

        for layer in self.layers.iter_mut() {
            match layer {
                LayerType::Activation(activation_layer) => {
                    if activation_layer.input_shape().len() == 2 {
                        flatten_input =
                            activation_layer.forward_propagate(&flatten_input, is_training);
                    } else {
                        input = activation_layer.forward_propagate(&input, is_training);
                    }
                }
                LayerType::BatchNormLayer(batchnorm_layer) => {
                    if batchnorm_layer.input_shape().len() == 2 {
                        let cache: Option<BNCache<Dim<[usize; 2]>>>;
                        (flatten_input, cache) =
                            batchnorm_layer.forward_propagate(flatten_input, is_training);
                        if let Some(cache) = cache {
                            self.store.add_bn2(cache);
                        }
                    } else {
                        let cache: Option<BNCache<Dim<[usize; 4]>>>;
                        (input, cache) = batchnorm_layer.forward_propagate(input, is_training);
                        if let Some(cache) = cache {
                            self.store.add_bn4(cache);
                        }
                    }
                }
                LayerType::Convolutional(convolutional_layer) => {
                    input = convolutional_layer.forward_propagate(input, is_training);
                }
                LayerType::Dense(dense_layer) => {
                    flatten_input = dense_layer.forward_propagate(flatten_input, is_training);
                }
                LayerType::DepthwiseConvLayer(depthwise_conv_layer) => {
                    input = depthwise_conv_layer.forward_propagate(input, is_training);
                }
                LayerType::Flatten(flatten_layer) => {
                    flatten_input = flatten_layer.forward_propagate(&input, is_training);
                }
                LayerType::GlobalAvgPooling(avg_pooling_layer) => {
                    flatten_input = avg_pooling_layer.forward_propagate(&input, is_training);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    input = max_pooling_layer.forward_propagate(&input, is_training);
                }
            };
        }
        flatten_input
    }

    fn backward_propagate(&mut self, mut error: Array2<f64>) {
        let cache_err = "No cache found for backpropagation calculations.";
        let mut shaped_error: Array4<f64> = Array4::zeros((0, 0, 0, 0));

        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerType::Activation(activation_layer) => {
                    if activation_layer.input_shape().len() == 2 {
                        error = activation_layer.backward_propagate(error, self.lr);
                    } else {
                        shaped_error = activation_layer.backward_propagate(shaped_error, self.lr);
                    }
                }
                LayerType::BatchNormLayer(batchnorm_layer) => {
                    if batchnorm_layer.input_shape().len() == 2 {
                        let cache = self.store.consume_bn2().expect(cache_err);
                        error = batchnorm_layer.backward_propagate(error, cache, self.lr);
                    } else {
                        let cache = self.store.consume_bn4().expect(cache_err);
                        shaped_error =
                            batchnorm_layer.backward_propagate(shaped_error, cache, self.lr);
                    }
                }
                LayerType::Convolutional(convolutional_layer) => {
                    shaped_error = convolutional_layer.backward_propagate(shaped_error, self.lr);
                }
                LayerType::Dense(dense_layer) => {
                    error = dense_layer.backward_propagate(error, self.lr);
                }
                LayerType::DepthwiseConvLayer(depthwise_conv_layer) => {
                    shaped_error = depthwise_conv_layer.backward_propagate(shaped_error, self.lr);
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
            }
        }
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
