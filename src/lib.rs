use cnn::{activation::Activation, CNN};
use config::{
    BATCH_EPSILON, CNN_OUTPUT_SIZE, DEFAULT_LEARNING_RATE, DROPOUT_RATE, INPUT_SHAPE,
    MINI_BATCH_SIZE, NORM_MOMENTUM, OUTPUT_DIR, SE_RATIO,
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
    let mut cnn = CNN::load_with_data(model, images.clone(), Some("0.0"));
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

pub fn train(model: Option<&str>, data: Dataset, epochs: usize, lr: Option<&str>) {
    let mut images = vec![];
    // Temporary
    for label in data.labels.iter().take(80) {
        let image = image_to_ndimage(get_image(label.image_path()));
        images.push(image);
    }

    let mut cnn = match model {
        Some(model) => CNN::load_with_data(model, images, lr),
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
    let mut cnn = cnn::CNN::new(MINI_BATCH_SIZE, epochs, images, DEFAULT_LEARNING_RATE);
    cnn.add_convolutional_layer(32, 3, 2, None, true);
    cnn.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
    cnn.add_activation_layer(Activation::SiLU);

    cnn.add_mbconv_layer(1, 16, 3, 1, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 24, 3, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 24, 3, 2, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 40, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 40, 5, 2, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 80, 3, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 80, 3, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 80, 3, 2, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 112, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 112, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 112, 5, 1, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 192, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 192, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 192, 5, 1, SE_RATIO, DROPOUT_RATE, true);
    cnn.add_mbconv_layer(6, 192, 5, 2, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_mbconv_layer(6, 320, 3, 1, SE_RATIO, DROPOUT_RATE, true);

    cnn.add_convolutional_layer(1280, 1, 1, None, true);
    cnn.add_batch_norm_layer(1, BATCH_EPSILON, NORM_MOMENTUM);
    cnn.add_activation_layer(Activation::SiLU);

    cnn.add_global_avg_pooling_layer();
    if DROPOUT_RATE > 0. {
        cnn.add_dropout_layer(DROPOUT_RATE);
    }
    cnn.add_dense_layer(CNN_OUTPUT_SIZE, 0.);
    cnn
}
