use clap::{Parser, Subcommand};
use face_reconstruction::dataset::Dataset;

#[derive(Parser)]
#[command(version, about, long_about=None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare dataset
    Prepare,

    /// Train existing or new model
    Train { model: Option<String> },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Prepare => {
            prepare_data();
        }
        Commands::Train { model } => {
            train_model(model.as_deref());
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
