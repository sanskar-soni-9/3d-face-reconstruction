use crate::config::{DATASET_INPUT_SIZE, DATA_DIR, INPUT_SHAPE};
use crate::utils::*;
use core::panic;
use rayon::prelude::*;
use serde::{ser::SerializeStruct, Serialize};
use std::fs;

pub struct Labels {
    image_path: String,
    color_para: [f64; 7],
    exp_para: [f64; 29],
    illum_para: [f64; 10],
    pose_para: [f64; 7],
    shape_para: [f64; 199],
    tex_para: [f64; 199],
    roi: [f64; 4],
    pt2d: [f64; 136],
    pts_2d: [f64; 136],
    pts_3d: [f64; 136],
}

impl Serialize for Labels {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut state = serializer.serialize_struct("Record", 11).unwrap();
        state.serialize_field("image_path", &self.image_path)?;
        state.serialize_field("color_para", &format!("{:?}", &self.color_para))?;
        state.serialize_field("exp_para", &format!("{:?}", &self.exp_para))?;
        state.serialize_field("illum_para", &format!("{:?}", &self.illum_para))?;
        state.serialize_field("pose_para", &format!("{:?}", &self.pose_para))?;
        state.serialize_field("shape_para", &format!("{:?}", &self.shape_para))?;
        state.serialize_field("tex_para", &format!("{:?}", &self.tex_para))?;
        state.serialize_field("roi", &format!("{:?}", &self.roi))?;
        state.serialize_field("pt2d", &format!("{:?}", &self.pt2d))?;
        state.serialize_field("pts_2d", &format!("{:?}", &self.pts_2d))?;
        state.serialize_field("pts_3d", &format!("{:?}", &self.pts_3d))?;
        state.end()
    }
}

impl Labels {
    pub fn image_path(&self) -> &str {
        &self.image_path
    }
    pub fn color_para(&self) -> [f64; 7] {
        self.color_para
    }
    pub fn exp_para(&self) -> [f64; 29] {
        self.exp_para
    }
    pub fn illum_para(&self) -> [f64; 10] {
        self.illum_para
    }
    pub fn pose_para(&self) -> [f64; 7] {
        self.pose_para
    }
    pub fn shape_para(&self) -> [f64; 199] {
        self.shape_para
    }
    pub fn tex_para(&self) -> [f64; 199] {
        self.tex_para
    }
    pub fn roi(&self) -> [f64; 4] {
        self.roi
    }
    pub fn pt2d(&self) -> [f64; 136] {
        self.pt2d
    }
    pub fn pts_2d(&self) -> [f64; 136] {
        self.pts_2d
    }
    pub fn pts_3d(&self) -> [f64; 136] {
        self.pts_3d
    }
}

pub struct Dataset {
    pub labels: Vec<Labels>,
}

impl Dataset {
    pub fn get_headers() -> Vec<String> {
        vec![
            "image_path".to_string(),
            "color_para".to_string(),
            "exp_para".to_string(),
            "illum_para".to_string(),
            "pose_para".to_string(),
            "shape_para".to_string(),
            "tex_para".to_string(),
            "roi".to_string(),
            "pt2d".to_string(),
            "pts_2d".to_string(),
            "pts_3d".to_string(),
        ]
    }
    pub fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let mut data_file = csv::Reader::from_path([DATA_DIR, "/afw.csv"].concat())?;
        let mut labels: Vec<Labels> = vec![];
        let mut image_path = String::new();
        let mut color_para = [0.0; 7];
        let mut exp_para = [0.0; 29];
        let mut illum_para = [0.0; 10];
        let mut pose_para = [0.0; 7];
        let mut shape_para = [0.0; 199];
        let mut tex_para = [0.0; 199];
        let mut roi = [0.0; 4];
        let mut pt2d = [0.0; 136];
        let mut pts_2d = [0.0; 136];
        let mut pts_3d = [0.0; 136];

        for record in data_file.records() {
            let record = record.unwrap();
            for (field, data) in Self::get_headers().iter().zip(record.iter()) {
                match field.as_str() {
                    "image_path" => image_path = data.to_string(),
                    "color_para" => color_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "exp_para" => exp_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "illum_para" => illum_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "pose_para" => pose_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "shape_para" => shape_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "tex_para" => tex_para = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "roi" => roi = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "pt2d" => pt2d = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "pts_2d" => pts_2d = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    "pts_3d" => pts_3d = vec_to_array_f64(&str_to_vec_f64(data).unwrap()),
                    _ => continue,
                }
            }
            labels.push(Labels {
                image_path: image_path.clone(),
                color_para,
                exp_para,
                illum_para,
                pose_para,
                shape_para,
                tex_para,
                roi,
                pt2d,
                pts_2d,
                pts_3d,
            })
        }

        Ok(Dataset { labels })
    }

    pub fn prepare() -> Self {
        let data = Self::get_afw_mat_data();
        let mut output_file = csv::WriterBuilder::new()
            .from_path([DATA_DIR, "/afw.csv"].concat())
            .expect("Error creating/writing afw.csv file");

        for record in &data.labels {
            output_file.serialize(record).unwrap();
        }
        output_file
            .flush()
            .expect("Error while saving csv file afw.csv.");

        data
    }

    fn get_afw_mat_data() -> Self {
        let img_dir_path = [DATA_DIR, "/AFW/"].concat();
        let img_dir = fs::read_dir(&img_dir_path).unwrap_or_else(|e| {
            panic!(
                "Error reading image directory: {}\nError: {}",
                &img_dir_path, e
            )
        });
        let mut records: Vec<Labels> = vec![];

        for file in img_dir {
            let dir_entry =
                file.unwrap_or_else(|e| panic!("Error getting image directory entry: {}", e));
            let entry_name = dir_entry.file_name();
            let Some(file_name) = entry_name.to_str() else {continue;};
            let Some(file_ext) = std::path::Path::new(&file_name).extension() else {continue;};
            if !file_ext.eq("jpg") {
                continue;
            }

            let image_labels = Self::get_labels_from_mat(
                &[
                    &[
                        DATA_DIR,
                        "/AFW/",
                        file_name.split_terminator('.').next().unwrap(),
                        ".mat",
                    ]
                    .concat(),
                    &[
                        DATA_DIR,
                        "/landmarks/AFW/",
                        file_name.split_terminator('.').next().unwrap(),
                        "_pts.mat",
                    ]
                    .concat(),
                ],
                [&img_dir_path, file_name].concat(),
            );

            records.push(image_labels);
        }

        Dataset { labels: records }
    }

    fn get_labels_from_mat(file_paths: &[&str], image_path: String) -> Labels {
        let mut color_para = [0.0; 7];
        let mut exp_para = [0.0; 29];
        let mut illum_para = [0.0; 10];
        let mut pose_para = [0.0; 7];
        let mut shape_para = [0.0; 199];
        let mut tex_para = [0.0; 199];
        let mut roi = [0.0; 4];
        let mut pt2d = [0.0; 136];
        let mut pts_2d = [0.0; 136];
        let mut pts_3d = [0.0; 136];

        for file_path in file_paths {
            let file = std::fs::File::open(file_path).unwrap_or_else(|e| {
                panic!("Error opening labels file: {}\nError:{}", file_path, e)
            });

            let mat_file = matfile::MatFile::parse(file)
                .unwrap_or_else(|e| panic!("Error parsing .mat file: {}\nError: {}", file_path, e));

            for array in mat_file.arrays() {
                match array.name().to_lowercase().as_str() {
                    "color_para" => {
                        color_para = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                vec_f32_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "exp_para" => {
                        exp_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "illum_para" => {
                        illum_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pose_para" => {
                        pose_para = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                vec_f32_to_array_f64(real)
                            }
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "shape_para" => {
                        shape_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "tex_para" => {
                        tex_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_array_f64(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "roi" => {
                        roi = match array.data() {
                            matfile::NumericData::Int16 { real, imag: _ } => {
                                Self::scale_labels(vec_i16_to_array_f64(real))
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pt2d" => {
                        pt2d = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                Self::scale_labels(vec_to_array_f64(real))
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pts_2d" => {
                        pts_2d = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                Self::scale_labels(vec_f32_to_array_f64(real))
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pts_3d" => {
                        pts_3d = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                Self::scale_labels(vec_f32_to_array_f64(real))
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    _ => (),
                }
            }
        }

        Labels {
            image_path,
            color_para,
            exp_para,
            illum_para,
            pose_para,
            shape_para,
            tex_para,
            roi,
            pt2d,
            pts_2d,
            pts_3d,
        }
    }

    fn scale_labels<const N: usize>(mut labels: [f64; N]) -> [f64; N] {
        let scale_factor = INPUT_SHAPE.1 as f64 / DATASET_INPUT_SIZE as f64;
        labels
            .par_iter_mut()
            .for_each(|label| *label *= scale_factor);
        labels
    }
}
