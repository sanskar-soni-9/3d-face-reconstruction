use core::panic;
use std::fs;

pub struct Afw {
    pub image_paths: Vec<String>,
    pub image_labels: Vec<Vec<u8>>,
}

impl Afw {
    #[must_use]
    pub fn new(data_dir_path: &str) -> Afw {
        // Get Training Data.
        let image_dir_path = [data_dir_path, "AFW/"].concat();
        println!("Reading MNIST training data. {}", data_dir_path);
        let (image_paths, image_labels) = parse_images_with_labels(&image_dir_path);

        Afw {
            image_paths,
            image_labels,
        }
    }
}

fn parse_images_with_labels(directory_path: &str) -> (Vec<String>, Vec<Vec<u8>>) {
    println!("At dir: {}", directory_path);
    let dir = match fs::read_dir(directory_path) {
        Ok(d) => d,
        Err(e) => panic!("Error reading directory: {}\n{}", directory_path, e),
    };
    let mut images: Vec<String> = vec![];
    let mut labels: Vec<Vec<u8>> = vec![];

    for file in dir {
        let dir_entry = match file {
            Ok(dir_entry) => dir_entry,
            Err(e) => panic!("Error getting directory entry: {}", e),
        };

        let entry_name = dir_entry.file_name();
        let Some(file_name) = entry_name.to_str() else {continue;};
        let Some(file_ext) = std::path::Path::new(&file_name).extension() else {continue;};
        if !file_ext.eq("jpg") {
            continue;
        }
        println!("File Found: {}{}", directory_path, file_name);

        let image_labels = parse_labels(
            &[
                directory_path,
                file_name.split_terminator('.').next().unwrap(),
                ".mat",
            ]
            .concat(),
        );

        images.push([directory_path, file_name].concat());
        labels.push(image_labels);
    }
    (images, labels)
}

fn parse_labels(directory_path: &str) -> Vec<u8> {
    // Yet to Implement
    let labels: Vec<u8> = vec![];
    labels
}
