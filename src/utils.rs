pub fn vec_into_f64_array<T: Into<f64> + Copy, const N: usize>(vec: &[T]) -> [f64; N] {
    if vec.len() != N {
        panic!("Unexpected vector length");
    }

    let mut array = [0.0; N];
    for i in 0..N {
        array[i] = vec[i].into();
    }
    array
}

pub fn vec_to_f64_array<const N: usize>(vec: &Vec<f64>) -> [f64; N] {
    vec.to_owned().try_into().unwrap_or_else(|v: Vec<f64>| {
        panic!("Expected a Vec of length {} but it was {}", N, v.len())
    })
}

pub fn vec_to_f32_array<const N: usize>(vec: &Vec<f32>) -> [f32; N] {
    vec.to_owned().try_into().unwrap_or_else(|v: Vec<f32>| {
        panic!("Expected a Vec of length {} but it was {}", N, v.len())
    })
}

pub fn vec_to_i16_array<const N: usize>(vec: &Vec<i16>) -> [i16; N] {
    vec.to_owned().try_into().unwrap_or_else(|v: Vec<i16>| {
        panic!("Expected a Vec of length {} but it was {}", N, v.len())
    })
}

pub fn str_to_vec_f32(s: &str) -> Result<Vec<f32>, std::num::ParseFloatError> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|x| x.trim().parse::<f32>())
        .collect()
}

pub fn str_to_vec_f64(s: &str) -> Result<Vec<f64>, std::num::ParseFloatError> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|x| x.trim().parse::<f64>())
        .collect()
}

pub fn str_to_vec_i16(s: &str) -> Result<Vec<i16>, std::num::ParseIntError> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(|x| x.trim().parse::<i16>())
        .collect()
}
