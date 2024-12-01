use cnn::{
    activation::Activation,
    optimizer::{OptimizerType, SgdmParameters},
    CNN,
};
use config::{
    BATCH_EPSILON, CNN_OUTPUT_SIZE, DEFAULT_LEARNING_RATE, INPUT_SHAPE, MINI_BATCH_SIZE,
    NORM_MOMENTUM, OUTPUT_DIR, SGD_MOMENTUM,
};
use dataset::Dataset;
use image::{imageops::FilterType, DynamicImage, GenericImage, GenericImageView, Rgba};
use ndarray::Array3;
use nshare::IntoNdarray3;

pub mod cnn;
pub mod config;
pub mod dataset;
pub mod utils;

pub fn infer(model: &str, image_paths: Vec<String>) {
    // TODO: temporary
    let images = get_ndimages(&image_paths);
    let mut cnn = CNN::load_with_data(model, images.clone());
    for (i, image) in image_paths.iter().enumerate() {
        let labels = cnn.infer(i);
        let mut new_img = get_image(image);
        for i in 0..labels.len() / 2 {
            let (x, y) = (labels[i] as u32, labels[(labels.len() / 2) + i] as u32);
            if x < INPUT_SHAPE.2 as u32 && y < INPUT_SHAPE.1 as u32 {
                new_img.put_pixel(x, y, Rgba([255, 255, 255, 1]));
            }
        }
        let image_path = format!("{}/img{}.png", OUTPUT_DIR, i);
        new_img.save(&image_path).unwrap_or_else(|e| {
            panic!(
                "Error occured while saving output file: {}\nError: {}",
                &image_path, e
            )
        });
    }
}

pub fn train(model: Option<&str>, data: Dataset, epochs: usize) {
    let mut images = vec![];
    // Temporary
    for label in data.labels.iter().take(80) {
        let image = image_to_ndimage(get_image(label.image_path()));
        images.push(image);
    }

    let mut cnn = match model {
        Some(model) => CNN::load_with_data(model, images),
        None => init_cnn(epochs, images),
    };
    cnn.train(data.labels);
}

pub fn get_image(image_path: &str) -> DynamicImage {
    let mut image = match image::open(image_path) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {}\nError: {}", image_path, message);
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

fn init_cnn(epochs: usize, images: Vec<Array3<f64>>) -> cnn::CNN {
    let mut cnn = cnn::CNN::new(
        MINI_BATCH_SIZE,
        epochs,
        images,
        OptimizerType::SgdMomentum(SgdmParameters {
            lr: DEFAULT_LEARNING_RATE,
            momentum: SGD_MOMENTUM,
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
