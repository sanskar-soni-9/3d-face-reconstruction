use face_reconstruction::dataset::Dataset;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        panic!("Invalid args...");
    }

    match args[1].as_str() {
        "prepare" => {
            prepare_data();
        }
        "train" => {
            train_model();
        }
        _ => {
            infer_model(args[1..2].to_vec()) // Currently only supporting 1
        }
    }
}

fn prepare_data() -> Dataset {
    Dataset::prepare()
}

fn infer_model(images: Vec<String>) {
    let images = face_reconstruction::get_images(&images);
    face_reconstruction::infer(images);
}

fn train_model() {
    let dataset = Dataset::load().unwrap_or_else(|_| {
        println!("CSV file not found, preparing dataset...");
        prepare_data()
    });
    face_reconstruction::train(dataset);
}
