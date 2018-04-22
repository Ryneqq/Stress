extern crate stress;
#[macro_use]
extern crate rulinalg;
use stress::NeuralNetwork;
use rulinalg::matrix::{Matrix, BaseMatrix};

fn main() {
    let mut nn = NeuralNetwork::new(vec![2, 2, 1]);
    let question = matrix![0., 1., 0., 1.; 0., 0., 1., 1.];
    let anwser   = matrix![0., 1., 1., 0.];
    nn.learn(question, anwser, 200).unwrap();
    println!("nn: {:?}", nn);
}
