// 300W-LP dataset: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP/main.htm
// homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

pub const DEFAULT_LEARNING_RATE: f64 = 0.256;
pub const MINI_BATCH_SIZE: usize = 4;
pub const BATCH_EPSILON: f64 = 1e-5;
pub const NORM_MOMENTUM: f64 = 0.99;
pub const SE_RATIO: f64 = 0.25; // Squeeze & Excite ratio
pub const DROPOUT_RATE: f64 = 0.2;

pub const INPUT_SHAPE: (usize, usize, usize) = (3, 224, 224); // (channles {3: RGB}, height, width)
pub const DATASET_INPUT_SIZE: usize = 450; // 450x450
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
pub const CONV_WEIGHT_SCALE: f64 = 2.0;
pub const DENSE_WEIGHT_SCALE: f64 = 1.0 / 3.0;

// Dirs & Paths
pub const OUTPUT_DIR: &str = "results";
pub const DATA_DIR: &str = "data";
pub const MODELS_DIR: &str = "models";
pub const ACCURATE_MODEL: &str = "accurate";
pub const NEWEST_MODEL: &str = "model";
pub const EPOCH_MODEL: &str = "epoch";
