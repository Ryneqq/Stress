use rand::{Rng, thread_rng};
use matrix::{Matrix, BaseMatrix};

type Mat = Matrix<f64>;

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    nn: Vec<Mat>
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>) -> Self {
        let nn: NN = layers.windows(2)
            .map(|layer| random_matrix(layer[0] + 1, layer[1]))
            .collect();
        NeuralNetwork{ nn }
    }

    pub fn run(&self, mut signal: Mat) -> Result<Mat, String> {
        self.nn.iter()
            .map(|layer| signal = activation_function(&(layer.transpose() * add_bias(&signal))))
            .next();

        let last_layer = self.nn.last().ok_or("Neural network does not contain any layer".to_string())?;
        Ok(activation_function(&(last_layer.transpose() * add_bias(&signal))))
    }

    pub fn learn(&mut self, input: Mat, output: Mat, reps: usize) -> Result<(), String> {
        let examples = number_of_examples(&input, &output)?;

        for _ in 0..reps {
            let example = random_example(&input, &output, examples);
            println!("which: {:?}", example);
        }

        Ok(())
    }

    // fn run_and_return_all_signals() {
    //     let mut singals: Vec<Mat> = Vec::new();
    //     self.nn.iter()
    //         .map(|layer| signal = NeuralNetwork::run_single_layer(layer, &signal))
    //         .map(|signal| singals.push(signal.clone()))
    //         .next();

    //     let last_layer = self.nn.last().ok_or("Neural network does not contain any layer".to_string())?;
    //     signal = activation_function(&(last_layer.transpose() * add_bias(&signal)));
    //     singals.push(signal);
    //     Ok(signals)
    // }

    fn run_single_layer(layer: &Mat, signal: &Mat) -> Mat {
        activation_function(&(layer.transpose() * add_bias(signal)))
    }
}

fn random_matrix(rows: usize, cols: usize) -> Mat {
    let mut rng = thread_rng();
    let content: Vec<f64> = (0..rows*cols).map(|_| rng.gen_range(-1.,1.)).collect();
 
    Matrix::new(rows,cols,content)
}

fn random_example(input: &Mat, output: &Mat, examples: usize) -> (Mat, Mat) {
    let mut rng = thread_rng();
    let which_example = rng.gen_range(0, examples);

    let input = input.col(which_example).into_matrix();
    let output = output.col(which_example).into_matrix();

    (input, output)
}

fn activation_function(network: &Mat) -> Mat {
    let beta = 5.;

    let activation: Vec<f64> = network.data()
        .iter()
        .map(|a| 1./(1. + (-beta*a).exp()))
        .collect();
    
    Matrix::new(network.rows(), network.cols(), activation)
}

fn add_bias(signal: &Mat) -> Mat {
    let mut signal_with_bias: Vec<f64> = vec![-1.];
    signal_with_bias.extend(signal.data().clone());

    Matrix::new(signal_with_bias.len(), 1, signal_with_bias)
}

fn remove_bias(signal: &Mat) -> Mat {
    let signal = signal.data().split_first().unwrap().1.to_vec();

    Matrix::new(signal.len(), 1, signal)
}

fn number_of_examples(input: &Mat, output: &Mat) -> Result<usize, String> {
    if input.cols() != output.cols() {
        return Err("Not consistent data".to_string());
    }
    Ok(input.cols())
}

#[cfg(test)]
mod tests {
    use super::*;

    mod methods {
        use super::*;
        #[test]
        fn run_test() {
            let nn: NN = vec![Matrix::ones(3,2), Matrix::ones(3,1)]; // 2-2-1 filled with ones
            let nn = NeuralNetwork { nn };

            let actuall = nn.run(matrix![0.;1.]).unwrap();
            let expected: Mat= Matrix::ones(1,1) / 2.;

            assert_eq!(expected, actuall);
        }

        // #[test]
        // fn learn_test() {
        //     let nn: NN = vec![Matrix::ones(3,2), Matrix::ones(3,1)]; // 2-2-1 filled with ones
        //     let nn = NeuralNetwork { nn };

        //     let actuall = nn.run(matrix![0.;1.]).unwrap();
        //     let expected: Mat= Matrix::ones(1,1) / 2.;

        //     assert_eq!(expected, actuall);
        // }
    }

    mod tools {
        use super::*;
        #[test]
        fn activation_function_test() {
            let matrix: Mat= Matrix::zeros(3,3);
            let actuall = activation_function(&matrix);
            let expected: Mat= Matrix::ones(3,3) / 2.;
            
            assert_eq!(actuall, expected);
        }

        #[test]
        fn add_bias_test() {
            let matrix: Mat = Matrix::zeros(3,1);
            let actuall = add_bias(&matrix);
            let expected: Mat = Matrix::new(4, 1, vec![-1.,0.,0.,0.]);
            
            assert_eq!(4, actuall.rows());
            assert_eq!(actuall, expected);
        }

        #[test]
        fn remove_bias_test() {
            let matrix: Mat = Matrix::new(4, 1, vec![-1.,0.,0.,0.]);
            let actuall = remove_bias(&matrix);
            let expected: Mat = Matrix::zeros(3,1);
            
            assert_eq!(3, actuall.rows());
            assert_eq!(actuall, expected);
        }

        #[test]
        fn random_matrix_test() {
            let random = random_matrix(5,5);
            let other = random_matrix(5,5);

            assert!(random != other, "This matricies should not be identical");
        }
    }
}