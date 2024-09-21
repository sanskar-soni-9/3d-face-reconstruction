use config::INPUT_SHAPE;
use dataset::Dataset;
use image::{imageops::FilterType, GenericImageView};
use ndarray::Array3;
use nshare::IntoNdarray3;

pub mod cnn;
pub mod config;
pub mod dataset;
pub mod utils;

pub fn infer(images: Vec<Array3<f32>>) {
    let mut cnn = init_cnn(0, images);
    // TODO: implement
}

pub fn train(data: Dataset) {
    let mut images = vec![];
    for (i, label) in data.labels.iter().enumerate() {
        // Temporary
        if i == 1 {
            break;
        }
        images.push(get_image(&label.image_path));
    }
    let mut cnn = init_cnn(1, images);
    cnn.train(data.labels);
}

pub fn get_image(image_path: &str) -> Array3<f32> {
    let mut image = match image::open(image_path) {
        Ok(image) => image,
        Err(message) => {
            println!("Failed to read image: {}", message);
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
    image.into_rgb32f().into_ndarray3()
}

pub fn get_images(image_paths: &[String]) -> Vec<Array3<f32>> {
    let mut images = vec![];
    for path in image_paths {
        images.push(get_image(path));
    }
    images
}

fn init_cnn(epochs: usize, images: Vec<Array3<f32>>) -> cnn::CNN {
    let mut cnn = cnn::CNN::new(epochs, images);
    cnn.add_convolutional_layer(16, 3, 1);
    cnn.add_max_pooling_layer(2, 1);
    cnn.add_convolutional_layer(32, 3, 1);
    cnn.add_max_pooling_layer(2, 1);
    cnn.add_convolutional_layer(64, 3, 1);
    cnn.add_max_pooling_layer(2, 1);
    cnn.add_convolutional_layer(128, 3, 1);
    cnn.add_max_pooling_layer(2, 1);
    cnn.add_flatten_layer();
    cnn
}
