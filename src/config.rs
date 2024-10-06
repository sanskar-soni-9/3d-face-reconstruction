// 300W-LP dataset: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP/main.htm
// homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

pub const DATASET_INPUT_SIZE: usize = 450; // 450x450
pub const INPUT_SHAPE: (usize, usize, usize) = (3, 450, 450); // (channles {3: RGB}, height, width)
pub const DEFAULT_LEARNING_RATE: f64 = 0.000_000_03;
pub const TRAINIG_LABELS: [&str; 1] = [
    "pts_2d", // 136
             // "shape_para", // 199
             // "pts_3d",     // 136
             // "tex_para",   // 199
             // "illum_para", // 10
             // "pose_para",  // 7
             // "exp_para",   // 29
             // "color_para", // 7
             // "pt2d",       // 136
             // "roi"?,       // 4
];
pub const CNN_OUTPUT_SIZE: usize = 136;

// Dirs & Paths
pub const OUTPUT_DIR: &str = "results";
pub const DATA_DIR: &str = "data";
pub const MODELS_DIR: &str = "models";
pub const ACCURATE_MODEL: &str = "accurate";
pub const NEWEST_MODEL: &str = "model";
pub const EPOCH_MODEL: &str = "epoch";
