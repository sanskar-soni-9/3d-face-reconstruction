use cnn::INPUT_SIZE;
use ndarray::Array3;
use nshare::IntoNdarray3;

pub mod cnn;
pub mod config;
pub mod dataset;
pub mod utils;

pub fn infer(images: Vec<Array3<f32>>) {
    let mut cnn = init_cnn(0, images);
    // Yet to Implement
}

pub fn train(images: Vec<Array3<f32>>) {
    let mut cnn = init_cnn(3, images);
    cnn.train(Array3::zeros(INPUT_SIZE));
}

pub fn get_images(image_paths: &[String]) -> Vec<Array3<f32>> {
    let mut images: Vec<Array3<f32>> = vec![];
    for image_path in image_paths {
        let image = match image::open(image_path) {
            Ok(image) => image,
            Err(message) => {
                println!("Failed to read image: {}", message);
                continue;
            }
        };
        images.push(
            image
                .clone()
                .resize_exact(
                    INPUT_SIZE.1 as u32,
                    INPUT_SIZE.2 as u32,
                    image::imageops::FilterType::Nearest,
                )
                .to_rgb32f()
                .into_ndarray3(),
        );
    }
    images
}

fn init_cnn(epochs: usize, images: Vec<Array3<f32>>) -> cnn::CNN {
    let mut cnn = cnn::CNN::new(epochs, images);
    cnn.add_convolutional_layer(16, 3, 1);
    cnn.add_max_polling_layer(2, 1);
    cnn.add_convolutional_layer(32, 3, 1);
    cnn.add_max_polling_layer(2, 1);
    cnn.add_convolutional_layer(64, 3, 1);
    cnn.add_max_polling_layer(2, 1);
    cnn.add_convolutional_layer(128, 3, 1);
    cnn.add_max_polling_layer(2, 1);
    cnn
}
