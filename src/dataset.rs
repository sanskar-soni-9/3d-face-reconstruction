use crate::config::DATA_DIR;
use crate::utils::*;
use core::{f64, panic};
use serde::{ser::SerializeStruct, Serialize};
use std::fs;

pub struct Labels {
    pub image_path: String,
    pub color_para: [f32; 7],
    pub exp_para: [f64; 29],
    pub illum_para: [f64; 10],
    pub pose_para: [f64; 7],
    pub shape_para: [f64; 199],
    pub tex_para: [f64; 199],
    pub roi: [i16; 4],
    pub pt2d: [f64; 136],
    pub pts_2d: [f32; 136],
    pub pts_3d: [f32; 136],
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
        let mut data_file = csv::Reader::from_path([DATA_DIR, "afw.csv"].concat())?;
        let mut labels: Vec<Labels> = vec![];
        let mut image_path = String::new();
        let mut color_para = [0.0; 7];
        let mut exp_para = [0.0; 29];
        let mut illum_para = [0.0; 10];
        let mut pose_para = [0.0; 7];
        let mut shape_para = [0.0; 199];
        let mut tex_para = [0.0; 199];
        let mut roi = [0; 4];
        let mut pt2d = [0.0; 136];
        let mut pts_2d = [0.0; 136];
        let mut pts_3d = [0.0; 136];

        for record in data_file.records() {
            let record = record.unwrap();
            for (field, data) in Self::get_headers().iter().zip(record.iter()) {
                match field.as_str() {
                    "image_path" => image_path = data.to_string(),
                    "color_para" => color_para = vec_to_f32_array(&str_to_vec_f32(data).unwrap()),
                    "exp_para" => exp_para = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "illum_para" => illum_para = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "pose_para" => pose_para = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "shape_para" => shape_para = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "tex_para" => tex_para = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "roi" => roi = vec_to_i16_array(&str_to_vec_i16(data).unwrap()),
                    "pt2d" => pt2d = vec_to_f64_array(&str_to_vec_f64(data).unwrap()),
                    "pts_2d" => pts_2d = vec_to_f32_array(&str_to_vec_f32(data).unwrap()),
                    "pts_3d" => pts_3d = vec_to_f32_array(&str_to_vec_f32(data).unwrap()),
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
            .from_path([DATA_DIR, "afw.csv"].concat())
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
        let img_dir_path = [DATA_DIR, "AFW/"].concat();
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
                        "AFW/",
                        file_name.split_terminator('.').next().unwrap(),
                        ".mat",
                    ]
                    .concat(),
                    &[
                        DATA_DIR,
                        "landmarks/AFW/",
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
        let mut roi = [0; 4];
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
                                vec_to_f32_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "exp_para" => {
                        exp_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "illum_para" => {
                        illum_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pose_para" => {
                        pose_para = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                vec_into_f64_array(real)
                            }
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "shape_para" => {
                        shape_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "tex_para" => {
                        tex_para = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "roi" => {
                        roi = match array.data() {
                            matfile::NumericData::Int16 { real, imag: _ } => vec_to_i16_array(real),
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pt2d" => {
                        pt2d = match array.data() {
                            matfile::NumericData::Double { real, imag: _ } => {
                                vec_to_f64_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pts_2d" => {
                        pts_2d = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                vec_to_f32_array(real)
                            }
                            _ => panic!("Unexpected Data"),
                        };
                    }
                    "pts_3d" => {
                        pts_3d = match array.data() {
                            matfile::NumericData::Single { real, imag: _ } => {
                                vec_to_f32_array(real)
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
}
