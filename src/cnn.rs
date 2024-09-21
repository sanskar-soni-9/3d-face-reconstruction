use crate::config::INPUT_SHAPE;
use crate::dataset::Labels;
use core::panic;
use layer::{convolutional_layer::*, flatten_layer::*, max_pooling_layer::*, *};
use ndarray::Array3;

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
                LayerType::Flatten(_) => {
                    panic!("Can't add convolutional layer after flatten layer.")
                }
                LayerType::Convolutional(layer) => {
                    ConvolutionalLayer::new(filters, kernel_size, strides, layer.output_size)
                }
                LayerType::MaxPooling(layer) => {
                    ConvolutionalLayer::new(filters, kernel_size, strides, layer.output_size)
                }
            },
            None => ConvolutionalLayer::new(filters, kernel_size, strides, INPUT_SHAPE),
        };

        self.add_layer(LayerType::Convolutional(layer));
    }

    pub fn add_max_pooling_layer(&mut self, kernel_size: usize, strides: usize) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }

        let layer = match self.layers.last() {
            None => panic!("Need Convolutional Layer Before Max Pooling Layer."),
            Some(layer) => match layer {
                LayerType::MaxPooling(_) => {
                    panic!("Need Convolutional Layer Before Max Pooling Layer.")
                }
                LayerType::Convolutional(layer) => layer,
                LayerType::Flatten(_) => {
                    panic!("Can't add max pooling layer after flatten layer.")
                }
            },
        };

        self.add_layer(LayerType::MaxPooling(MaxPoolingLayer::new(
            kernel_size,
            layer.output_size,
            strides,
        )));
    }

    pub fn add_flatten_layer(&mut self) {
        self.add_layer(LayerType::Flatten(FlattenLayer::new()));
    }

    pub fn train(&mut self, labels: Vec<Labels>) {
        // TODO: implement
        for _ in 0..self.epochs {
            for (i, _) in labels.iter().enumerate().take(self.data.len()) {
                self.forward_propagate(self.data[i].clone());
            }
        }
    }

    fn forward_propagate(&mut self, image: Array3<f32>) {
        let mut output = image;
        let mut flatten_output;
        for layer in self.layers.iter_mut() {
            match layer {
                LayerType::Convolutional(convolutional_layer) => {
                    output = convolutional_layer.forward_propagate(&output);
                }
                LayerType::MaxPooling(max_pooling_layer) => {
                    output = max_pooling_layer.forward_propagate(&output);
                }
                LayerType::Flatten(flatten_layer) => {
                    flatten_output = flatten_layer.forward_propagate(&output);
                }
            };
        }
    }

    fn backward_propagate(&self, error: &Labels) {
        // TODO: implement
    }

    fn add_layer(&mut self, layer: LayerType) {
        self.layers.push(layer);
    }
}
