use face_reconstruction::{afw::Afw, config::DEFAULT_DATA_DIR};
use std::env::Args;

fn main() {
    parse_args(std::env::args());
}

fn infer_model(images: Vec<String>) {
    let images = face_reconstruction::get_images(&images);
    face_reconstruction::infer(images);
}

fn train_model(images: Vec<String>) {
    let images = face_reconstruction::get_images(&images);
    face_reconstruction::train(images);
}

fn parse_args(args: Args) {
    let args: Vec<String> = args.into_iter().collect();
    println!("{}", args.len());
    if args.len() < 2 {
        panic!("Invalid args...");
    }

    if args[1] == "train" {
        let path: String = args
            .get(2)
            .cloned()
            .unwrap_or_else(|| DEFAULT_DATA_DIR.to_string());
        let afw = Afw::new(&path);
        train_model(afw.image_paths);
        return;
    }

    infer_model(args[1..2].to_vec()); // Currently only supporting 1
}
