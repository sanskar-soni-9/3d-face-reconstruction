use crate::{dataset::Labels, get_image, image_to_ndimage, INPUT_SHAPE};
use ndarray::{s, Array2, Array4};
use rand::{rngs::ThreadRng, seq::SliceRandom, thread_rng};

pub struct DataLoader<'a> {
    rng: ThreadRng,
    cur_batch: usize,
    batch_size: usize,
    train_labels_size: usize,
    labels: &'a mut [Labels],
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new(batch_size: usize, shuffle: bool, labels: &'a mut [Labels]) -> Self {
        let mut rng = thread_rng();
        if shuffle {
            labels.shuffle(&mut rng);
        }
        DataLoader {
            rng,
            cur_batch: 0,
            batch_size,
            train_labels_size: labels
                .first()
                .expect("Labels vector cannot be empty in DataLoader.")
                .training_size(),
            labels,
            shuffle,
        }
    }

    pub fn reset_iter(&mut self) {
        self.cur_batch = 0;
        if self.shuffle {
            self.labels.shuffle(&mut self.rng);
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Array4<f64>, Array2<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cur_batch * self.batch_size >= self.labels.len() {
            return None;
        }

        let mut images: Array4<f64> =
            Array4::zeros((self.batch_size, INPUT_SHAPE.0, INPUT_SHAPE.1, INPUT_SHAPE.2));
        let mut labels: Array2<f64> = Array2::zeros((self.batch_size, self.train_labels_size));

        for batch in 0..self.batch_size {
            let idx = self.cur_batch * self.batch_size + batch;
            if idx < self.labels.len() {
                images
                    .slice_mut(s![batch, .., .., ..])
                    .assign(&image_to_ndimage(get_image(self.labels[idx].image_path())));
                labels
                    .slice_mut(s![batch, ..])
                    .assign(&self.labels[idx].as_training_array());
            } else {
                let label = self.labels.choose(&mut self.rng).unwrap();

                images.slice_mut(s![batch, .., .., ..]);
                labels
                    .slice_mut(s![batch, ..])
                    .assign(&label.as_training_array());
            }
        }

        self.cur_batch += 1;
        Some((images, labels))
    }
}
