// 300W-LP dataset: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP/main.htm
// homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

/// Model configurations
// General
pub const CONV_WEIGHT_SCALE: f64 = 2.;
pub const DENSE_WEIGHT_SCALE: f64 = 2.;
pub const BATCH_EPSILON: f64 = 1.001e-5;

// EfficientNet
pub const SE_RATIO: f64 = 0.25; // Squeeze & Excite ratio

/// Trainig configurations
// General
pub const DEFAULT_LEARNING_RATE: f64 = 0.0001;
pub const MINI_BATCH_SIZE: usize = 3;
pub const DROPOUT_RATE: f64 = 0.2;
pub const NORM_MOMENTUM: f64 = 0.99;
pub const TRAINING_DATA_PERCENT: f64 = 0.8;
pub const TOTAL_EPOCHS: usize = 200;

// Adam
pub const ADAM_BETA_1: f64 = 0.9;
pub const ADAM_BETA_2: f64 = 0.999;
pub const ADAM_EPSILON: f64 = 1e-8;
pub const ADAM_USE_AMS_GRAD: bool = false;

// Sgd with momentum
pub const SGD_MOMENTUM: f64 = 0.9;

/// Training labels
pub const TRAINIG_LABELS: [&str; 1] = [
    "pts_2d", // 136
             // "pt2d",       // 136
             // "pts_3d",     // 136
             // "exp_para",   // 29
             // "illum_para", // 10
             // "pose_para",  // 7
             // "color_para", // 7
             // "shape_para", // 199
             // "tex_para",   // 199
             // "roi",        // 4
];
pub const CNN_OUTPUT_SIZE: usize = 136;

/// Miscellaneous configurations
pub const PRETTY_SAVE: bool = false;

/// Model inputs configurations
pub const INPUT_SHAPE: (usize, usize, usize) = (3, 224, 224); // (channles {3: RGB}, height, width)
pub const DATASET_INPUT_SIZE: usize = 450; // 450x450

// Dirs & Files
pub const OUTPUT_DIR: &str = "results";
pub const DATA_DIR: &str = "data";
pub const MODELS_DIR: &str = "models";
pub const ACCURATE_MODEL: &str = "accurate";
pub const NEWEST_MODEL: &str = "model";
pub const EPOCH_MODEL: &str = "epoch";
