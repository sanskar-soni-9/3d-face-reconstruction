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
            let mut model: Option<&str> = None;
            let mut lr: Option<&str> = None;
            if args.len() >= 4 {
                lr = Some(args[3].as_str());
            }
            if args.len() >= 3 {
                model = Some(args[2].as_str());
            }
            train_model(model, lr);
        }
        _ => {
            println!("ARGS: {:?}", args);
            let mut model = None;
            if args.len() >= 3 {
                model = Some(args[2].as_str());
            }
            infer_model(model, args[3..].to_vec());
        }
    }
}

fn prepare_data() -> Dataset {
    Dataset::prepare()
}

fn infer_model(model: Option<&str>, images: Vec<String>) {
    face_reconstruction::infer(model, images);
}

fn train_model(model: Option<&str>, lr: Option<&str>) {
    let dataset = Dataset::load().unwrap_or_else(|_| {
        println!("CSV file not found, preparing dataset...");
        prepare_data()
    });
    face_reconstruction::train(model, dataset, 10, lr);
}
