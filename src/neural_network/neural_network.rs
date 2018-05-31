use rand::{Rng, thread_rng};
use matrix::{Matrix, BaseMatrix, BaseMatrixMut};

type Mat    = Matrix<f64>;
type Vector = Vec<f64>;

const BETA: f64 = 5.0;

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralNetwork {
    nn: Vec<Mat>
}

impl NeuralNetwork {
    pub fn new(layers: Vec<usize>) -> Self {
        let nn: Vec<Mat> = layers.windows(2)
            .map(|layer| random_matrix(layer[0] + 1, layer[1]))
            .collect();
        NeuralNetwork{ nn }
    }

    pub fn run(&self, signal: Vector) -> Result<Vector, String> {
        let mut signal = Matrix::new(signal.len(), 1, signal);
        self.nn.iter()
            .map(|layer| signal = Self::run_single_layer(layer, &signal))
            .next();

        let last_layer = self.nn.last().ok_or("Neural network does not contain any layer".to_string())?;
        Ok(Self::run_single_layer(last_layer, &signal).into_vec())
    }

    pub fn learn(&mut self, input: Mat, output: Mat, reps: usize) -> Result<(), String> { //not tested
        let examples = number_of_examples(&input, &output)?;

        for _ in 0..reps {
            let example = random_example(&input, &output, examples);
            let signals = self.run_and_return_all_signals(&example.0)?;
            let delta   = example.1 - signals.last().ok_or("No signals".to_string())?;
            self.back_propagation(signals, delta);
        }

        Ok(())
    }

    fn back_propagation(&mut self, signals: Vec<Mat>, delta: Mat) { //not tested
        let learning_rate  = 0.05;
        let mut delta      = delta.clone();
        let mut signals    = signals.clone();

        self.nn.iter_mut().rev()
            .map(|layer| {
                let signal = signals.pop().expect("No more signals");
                let error  = Self::find_error(&signal, &delta);
                let dnn    = error.transpose() * signals.last().expect("No more signals") * learning_rate;
                *layer     = layer.clone() - dnn;
                delta      = remove_bias(&layer) * error;
            }).next();
    }

    fn find_error(signal: &Mat, delta: &Mat) -> Mat {
        let error = Matrix::ones(signal.rows(), signal.cols()) - signal;

        error.elemul(&signal.elemul(&delta)) * BETA
    }

    fn run_and_return_all_signals(&self, signal: &Mat) -> Result<Vec<Mat>, String> {
        let mut signal = signal.clone();
        let mut signals: Vec<Mat> = vec![signal.clone()];

        self.nn.iter()
            .map(|layer| {
                signal = Self::run_single_layer(layer, &signal);
                signals.push(signal.clone())
            }).next();

        let last_layer = self.nn.last().ok_or("Neural network does not contain any layer".to_string())?;
        signals.push(Self::run_single_layer(last_layer, &signal));

        Ok(signals)
    }

    fn run_single_layer(layer: &Mat, signal: &Mat) -> Mat {
        activation_function(layer.transpose() * add_bias(signal))
    }
}

fn random_matrix(rows: usize, cols: usize) -> Mat {
    let mut rng = thread_rng();
    let content: Vector = (0..rows*cols).map(|_| rng.gen_range(-1.,1.)).collect();
  
    Matrix::new(rows,cols,content)
}

fn random_example(input: &Mat, output: &Mat, examples: usize) -> (Mat, Mat) { // not tested
    let mut rng = thread_rng();
    let which_example = rng.gen_range(0, examples);

    let input = input.col(which_example).into_matrix();
    let output = output.col(which_example).into_matrix();

    (input, output)
}

fn activation_function(network: Mat) -> Mat {
    let activation: Vector = network.data()
        .iter()
        .map(|a| 1.0/(1.0 + (-BETA*a).exp()))
        .collect();
    
    Matrix::new(network.rows(), network.cols(), activation)
}

fn add_bias(signal: &Mat) -> Mat {
    let mut signal_with_bias: Vector = vec![-1.];
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
            let nn: Vec<Mat> = vec![Matrix::ones(3,2), Matrix::ones(3,1)]; // 2-2-1 filled with ones
            let nn = NeuralNetwork { nn };

            let actual = nn.run(vec![0.0, 1.0]).unwrap();
            let expected: Vector = vec![0.5];

            assert_eq!(expected, actual);
        }

        #[test]
        fn run_and_return_all_signals_test() {
            let nn: Vec<Mat> = vec![Matrix::ones(3,2), Matrix::ones(3,1)]; // 2-2-1 filled with ones
            let nn = NeuralNetwork { nn };

            let actual = nn.run_and_return_all_signals(&matrix![0.0; 1.0]).unwrap();
            let expected = vec![matrix![0.0; 1.0], matrix![0.5; 0.5], matrix![0.5]];

            assert_eq!(expected, actual);
        }
    }

    mod tools {
        use super::*;
        #[test]
        fn activation_function_test() {
            let matrix: Mat= Matrix::zeros(3,3);
            let actual = activation_function(matrix);
            let expected: Mat= Matrix::ones(3,3) / 2.;
            
            assert_eq!(actual, expected);
        }

        #[test]
        fn add_bias_test() {
            let matrix: Mat = Matrix::zeros(3,1);
            let actual = add_bias(&matrix);
            let expected: Mat = Matrix::new(4, 1, vec![-1.0, 0.0, 0.0, 0.0]);
            
            assert_eq!(4, actual.rows());
            assert_eq!(actual, expected);
        }

        #[test]
        fn remove_bias_test() {
            let matrix: Mat = Matrix::new(4, 1, vec![-1.0, 0.0, 0.0, 0.0]);
            let actual = remove_bias(&matrix);
            let expected: Mat = Matrix::zeros(3,1);
            
            assert_eq!(3, actual.rows());
            assert_eq!(actual, expected);
        }

        #[test]
        fn random_matrix_test() {
            let random = random_matrix(5,5);
            let other = random_matrix(5,5);

            assert!(random != other, "This matricies should not be identical");
        }
    }
}