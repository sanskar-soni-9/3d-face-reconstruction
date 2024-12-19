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
            if args.len() >= 3 {
                model = Some(args[2].as_str());
            }
            train_model(model);
        }
        _ => {
            panic!("Invalid args...");
        }
    }
}

fn prepare_data() -> Dataset {
    Dataset::prepare()
}

fn train_model(model: Option<&str>) {
    let dataset = Dataset::load().unwrap_or_else(|_| {
        println!("CSV file not found, preparing dataset...");
        prepare_data()
    });
    face_reconstruction::train(model, dataset);
}
