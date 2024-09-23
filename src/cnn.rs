use crate::config::{INPUT_SHAPE, TRAINIG_LABELS};
use crate::dataset::Labels;
use dense_layer::DenseLayer;
use layer::{convolutional_layer::*, flatten_layer::*, max_pooling_layer::*, *};
use ndarray::{Array1, Array3};

mod layer;

pub struct CNN {
    layers: Vec<layer::LayerType>,
    epochs: usize,
    data: Vec<Array3<f32>>,
}

impl CNN {
    pub fn new(epochs: usize, inputs: Vec<Array3<f32>>) -> Self {
        CNN {
            layers: vec![],
            epochs,
            data: inputs,
        }
    }

    pub fn add_convolutional_layer(&mut self, filters: usize, kernel_size: usize, strides: usize) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let layer = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Convolutional(layer) => {
                    ConvolutionalLayer::new(filters, kernel_size, strides, layer.output_size)
                }
                LayerType::MaxPooling(layer) => {
                    ConvolutionalLayer::new(filters, kernel_size, strides, layer.output_size)
                }
                _ => panic!("Add convolutional layer after a convolutional or max pool layer."),
            },
            None => ConvolutionalLayer::new(filters, kernel_size, strides, INPUT_SHAPE),
        };

        self.add_layer(LayerType::Convolutional(layer));
    }

    pub fn add_max_pooling_layer(&mut self, kernel_size: usize, strides: usize) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }
        let output_size = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Convolutional(layer) => layer.output_size,
                LayerType::MaxPooling(layer) => layer.output_size,
                _ => panic!("Add max pooling layer after a convolutional or max pooling layer."),
            },
            None => panic!("Add max pooling layer after a convolutional or max pooling layer."),
        };

        self.add_layer(LayerType::MaxPooling(MaxPoolingLayer::new(
            kernel_size,
            output_size,
            strides,
        )));
    }

    pub fn add_flatten_layer(&mut self) {
        let input_size = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Convolutional(layer) => layer.output_size,
                LayerType::MaxPooling(layer) => layer.output_size,
                _ => panic!("Add flatten layer after a convolutional or max pooling layer."),
            },
            None => panic!("Add flatten layer after a convolutional or max pooling layer."),
        };
        self.add_layer(LayerType::Flatten(FlattenLayer::new(input_size)));
    }

    pub fn add_dense_layer(&mut self, neurons: usize, bias: f32, dropout_rate: f32) {
        let input_size = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::Dense(layer) => layer.output_size,
                LayerType::Flatten(layer) => {
                    layer.input_size.0 * layer.input_size.1 * layer.input_size.2
                }
                _ => panic!("Add dense layer after a flatten or dense layer."),
            },
            None => panic!("Add dense layer after a flatten or dense layer."),
        };

        self.add_layer(LayerType::Dense(DenseLayer::new(
            input_size,
            neurons,
            bias,
            dropout_rate,
        )));
    }

    pub fn train(&mut self, labels: Vec<Labels>) {
        // TODO: implement
        for _ in 0..self.epochs {
            for (i, labels) in labels.iter().enumerate().take(self.data.len()) {
                let training_labels = self.prepare_training_labels(labels);
                self.forward_propagate(self.data[i].clone());
                self.backward_propagate(training_labels);
            }
        }
    }

    fn forward_propagate(&mut self, image: Array3<f32>) {
        let mut output = image;
        let mut flatten_output = Array1::zeros(0);
        for layer in self.layers.iter_mut() {
            match layer {
                LayerType::Convolutional(convolutional_layer) => {
                    output = convolutional_layer.forward_propagate(&output, true);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    output = max_pooling_layer.forward_propagate(&output, true);
                }
                LayerType::Flatten(flatten_layer) => {
                    flatten_output = flatten_layer.forward_propagate(&output, true);
                }
                LayerType::Dense(dense_layer) => {
                    flatten_output = dense_layer.forward_propagate(&flatten_output, true);
                }
            };
        }
    }

    fn backward_propagate(&mut self, mut error: Vec<f32>) {
        // TODO: implement
        for layer in self.layers.iter_mut().rev() {
            match layer {
                LayerType::Convolutional(convolutional_layer) => {
                    error = convolutional_layer.backward_propagate(&error);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    error = max_pooling_layer.backward_propagate(&error);
                }
                LayerType::Flatten(_) => continue,
                LayerType::Dense(dense_layer) => {
                    error = dense_layer.backward_propagate(&error);
                }
            }
        }
    }

    fn prepare_training_labels(&self, labels: &Labels) -> Vec<f32> {
        let mut training_labels: Vec<f32> = vec![];
        for train_label in TRAINIG_LABELS {
            match train_label {
                "pts_2d" => training_labels.append(&mut labels.pts_2d.to_vec()),
                "pts_3d" => training_labels.append(&mut labels.pts_3d.to_vec()),
                "pose_para" => training_labels.append(&mut labels.pose_para.to_vec()),
                "shape_para" => training_labels.append(&mut labels.shape_para.to_vec()),
                "illum_para" => training_labels.append(&mut labels.illum_para.to_vec()),
                "color_para" => training_labels.append(&mut labels.color_para.to_vec()),
                "exp_para" => training_labels.append(&mut labels.exp_para.to_vec()),
                "tex_para" => training_labels.append(&mut labels.tex_para.to_vec()),
                "pt2d" => training_labels.append(&mut labels.pts_2d.to_vec()),
                _ => panic!("Unknown Label"),
            }
        }
        training_labels
    }

    fn add_layer(&mut self, layer: LayerType) {
        self.layers.push(layer);
    }
}
