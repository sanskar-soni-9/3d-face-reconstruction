use super::BNCache;
use ndarray::Dim;

type BN2 = BNCache<Dim<[usize; 2]>>;
type BN4 = BNCache<Dim<[usize; 4]>>;

#[derive(Default)]
pub struct CNNCache {
    bn2d_store: Vec<BN2>,
    bn4d_store: Vec<BN4>,
}

impl CNNCache {
    pub fn clear(&mut self) {
        self.bn2d_store.clear();
        self.bn4d_store.clear();
    }

    pub fn add_bn2(&mut self, value: BN2) {
        self.bn2d_store.push(value);
    }

    pub fn consume_bn2(&mut self) -> Option<BN2> {
        self.bn2d_store.pop()
    }

    pub fn add_bn4(&mut self, value: BN4) {
        self.bn4d_store.push(value);
    }

    pub fn consume_bn4(&mut self) -> Option<BN4> {
        self.bn4d_store.pop()
    }
}
