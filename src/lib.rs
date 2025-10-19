use cnn::{
    activation::Activation,
    optimizer::{AdamParameters, OptimizerType},
    CNN,
};
use config::*;
use dataset::Dataset;
use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::Array3;
use nshare::IntoNdarray3;

pub mod cnn;
pub mod config;
pub mod data_loader;
pub mod dataset;
pub mod utils;

pub fn train(model: Option<&str>, data: Dataset) {
    let mut cnn = match model {
        Some(model) => CNN::load(model),
        None => init_cnn(),
    };
    cnn.train(data.labels);
}

pub fn get_image(image_path: &str) -> DynamicImage {
    let mut image = match image::open(image_path) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {image_path}\nError: {message}");
            panic!("Error opnening image.");
        }
    };
    if image.dimensions() != (INPUT_SHAPE.1 as u32, INPUT_SHAPE.2 as u32) {
        image = image.resize_exact(
            INPUT_SHAPE.1 as u32,
            INPUT_SHAPE.2 as u32,
            FilterType::Lanczos3,
        );
    }
    image
}

pub fn image_to_ndimage(image: DynamicImage) -> Array3<f64> {
    image.into_rgb32f().into_ndarray3().mapv(|x| x as f64)
}

pub fn get_ndimages(image_paths: &[String]) -> Vec<Array3<f64>> {
    let mut images = vec![];
    for path in image_paths {
        images.push(image_to_ndimage(get_image(path)));
    }
    images
}

fn init_cnn() -> cnn::CNN {
    let mut cnn = cnn::CNN::new(
        DEFAULT_LEARNING_RATE,
        MINI_BATCH_SIZE,
        TOTAL_EPOCHS,
        Some(TRAINING_DATA_PERCENT),
        OptimizerType::Adam(AdamParameters {
            beta_1: ADAM_BETA_1,
            beta_2: ADAM_BETA_2,
            epsilon: ADAM_EPSILON,
            ams_grad: ADAM_USE_AMS_GRAD,
        }),
    );

    // ResNetV2
    cnn.add_convolutional_layer(64, 7, 2, Some(0.), true);
    cnn.add_max_pooling_layer(3, 2, true);

    cnn.stack_resnet_v2_blocks(64, 2, 3);
    cnn.stack_resnet_v2_blocks(128, 2, 4);
    cnn.stack_resnet_v2_blocks(256, 2, 6);
    cnn.stack_resnet_v2_blocks(512, 1, 3);

    cnn.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
    cnn.add_activation_layer(Activation::ReLU);

    cnn.add_global_avg_pooling_layer();
    cnn.add_dense_layer(CNN_OUTPUT_SIZE, 0.);

    cnn
}
