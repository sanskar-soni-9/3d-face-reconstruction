use super::{BNCache, OperandCache};
use ndarray::Dim;
use std::collections::HashMap;

type Dim2 = Dim<[usize; 2]>;
type Dim4 = Dim<[usize; 4]>;
type BN2 = BNCache<Dim2>;
type BN4 = BNCache<Dim4>;
type Layer2 = OperandCache<Dim2>;
type Layer4 = OperandCache<Dim4>;

#[derive(Default)]
pub struct CNNCache {
    bn2d_store: Vec<BN2>,
    bn4d_store: Vec<BN4>,
    operand2d_store: HashMap<usize, Layer2>,
    operand4d_store: HashMap<usize, Layer4>,
}

impl CNNCache {
    pub fn clear(&mut self) {
        self.bn2d_store.clear();
        self.bn4d_store.clear();
        self.operand2d_store.clear();
        self.operand2d_store.clear();
    }

    // BNCache
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

    // OperandCache
    pub fn add_operand2d(&mut self, id: usize, value: Layer2) {
        self.operand2d_store.insert(id, value);
    }
    pub fn get_operand2d(&self, id: usize) -> Option<&Layer2> {
        self.operand2d_store.get(&id)
    }
    pub fn consume_operand2d(&mut self, id: usize) -> Option<(usize, Layer2)> {
        self.operand2d_store.remove_entry(&id)
    }

    pub fn add_operand4d(&mut self, id: usize, value: Layer4) {
        self.operand4d_store.insert(id, value);
    }
    pub fn get_operand4d(&self, id: usize) -> Option<&Layer2> {
        self.operand2d_store.get(&id)
    }
    pub fn consume_operand4d(&mut self, id: usize) -> Option<(usize, Layer4)> {
        self.operand4d_store.remove_entry(&id)
    }
}
