use core::panic;
use layer::LayerType;
use layer::{convolutional_layer::*, max_polling_layer::*};
use ndarray::Array3;

mod layer;

pub const INPUT_SIZE: (usize, usize, usize) = (3, 400, 500); // Temporary

pub struct CNN {
    layers: Vec<layer::LayerType>,
    epochs: usize,
    data: Vec<Array3<f32>>,
    labels: Array3<f32>,
}

impl CNN {
    pub fn new(epochs: usize, data: Vec<Array3<f32>>) -> Self {
        CNN {
            layers: vec![],
            epochs,
            data,
            labels: Array3::zeros(INPUT_SIZE),
        }
    }

    pub fn add_convolutional_layer(&mut self, filters: usize, kernel_size: usize, strides: usize) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }

        let layer = match self.layers.last() {
            Some(layer) => match layer {
                LayerType::ConvolutionalLayer(_) => {
                    panic!("Can't add another convolutional layer after one.")
                }
                LayerType::MaxPollingLayer(layer) => {
                    ConvolutionalLayer::new(filters, kernel_size, strides, layer.output_size)
                }
            },
            None => ConvolutionalLayer::new(filters, kernel_size, strides, INPUT_SIZE),
        };

        self.add_layer(LayerType::ConvolutionalLayer(layer));
    }

    pub fn add_max_polling_layer(&mut self, kernel_size: usize, strides: usize) {
        if strides == 0 {
            panic!("Stride should be greater than 0.");
        }

        let layer = match self.layers.last() {
            None => panic!("Need Convolutional Layer Before Max Polling Layer."),
            Some(layer) => match layer {
                LayerType::MaxPollingLayer(_) => {
                    panic!("Need Convolutional Layer Before Max Polling Layer.")
                }
                LayerType::ConvolutionalLayer(layer) => layer,
            },
        };

        self.add_layer(LayerType::MaxPollingLayer(MaxPollingLayer::new(
            kernel_size,
            layer.output_size,
            strides,
        )));
    }

    pub fn train(&mut self, labels: Array3<f32>) {
        self.labels = labels;
        // TODO: implement
    }

    fn add_layer(&mut self, layer: LayerType) {
        self.layers.push(layer);
    }
}
