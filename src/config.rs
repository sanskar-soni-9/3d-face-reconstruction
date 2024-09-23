// 300W-LP dataset: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP/main.htm
// homepage: http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm

pub const DATA_DIR: &str = "data/";

pub const INPUT_SHAPE: (usize, usize, usize) = (3, 450, 450); // (channles {3: RGB}, width, height)

pub const TRAINIG_LABELS: [&str; 6] = [
    "pts_2d",
    "pts_3d",
    "pose_para",
    "shape_para",
    "illum_para",
    "color_para",
    // "exp_para",
    // "tex_para",
    // "pt2d",
    // "roi"?,
];

pub const CNN_OUTPUT_SIZE: usize = 495;
