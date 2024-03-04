/*!
Provides implementations of and related to Discrete Bayes filtering.
*/
use alloc::vec::Vec;
use num_traits::Float;

use crate::common::vec::{convolve, ConvolutionMode};
use crate::common::vec::{shift, ShiftMode};

/// Normalize distribution `pdf` in-place so it sums to 1.0.
///
/// # Example
///
/// ```
/// use kalmanfilt::discrete_bayes::normalize;
/// use assert_approx_eq::assert_approx_eq;
///
/// let mut pdf = [1.0, 1.0, 1.0, 1.0];
/// normalize(&mut pdf);
///
/// assert_approx_eq!(pdf[0], 0.25_f64);
/// assert_approx_eq!(pdf[1], 0.25_f64);
/// assert_approx_eq!(pdf[2], 0.25_f64);
/// assert_approx_eq!(pdf[3], 0.25_f64);
/// ```
///
pub fn normalize<F: Float>(pdf: &mut [F]) {
    let sum = pdf.iter().fold(F::zero(), |p, q| p + *q);
    pdf.iter_mut().for_each(|f| *f = *f / sum);
}

/// An error that can occur when updating a discrete Bayes filter.
#[derive(Debug, Copy, Clone)]
pub enum UpdateError {
    /// The likelihood and prior have different lengths.
    LengthMismatch,
}

/// Computes the posterior of a discrete random variable given a
/// discrete likelihood and prior. In a typical application the likelihood
/// will be the likelihood of a measurement matching your current environment,
/// and the prior comes from discrete_bayes.predict().
///
pub fn update<F: Float>(likelihood: &[F], prior: &[F]) -> Result<Vec<F>, UpdateError> {
    if likelihood.len() != prior.len() {
        return Err(UpdateError::LengthMismatch);
    }
    let mut posterior: Vec<F> = likelihood
        .iter()
        .zip(prior.iter())
        .map(|(&l, &p)| l * p)
        .collect();
    normalize(&mut posterior);
    Ok(posterior)
}

/// Determines what happens at the boundaries of the probability distribution.
#[derive(Debug)]
pub enum EdgeHandling<F> {
    /// the  probability distribution is shifted and the given value is used to used to fill in missing elements.
    Constant(F),
    /// The probability distribution is wrapped around the array.
    Wrap,
}

/// Performs the discrete Bayes filter prediction step, generating the prior.
pub fn predict<F: Float>(pdf: &[F], offset: i64, kernel: &[F], mode: EdgeHandling<F>) -> Vec<F> {
    match mode {
        EdgeHandling::Constant(c) => convolve(
            &shift(pdf, offset, ShiftMode::Extend(c)),
            kernel,
            ConvolutionMode::Extended(c),
        ),
        EdgeHandling::Wrap => convolve(
            &shift(pdf, offset, ShiftMode::Wrap),
            kernel,
            ConvolutionMode::Wrap,
        ),
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use assert_approx_eq::assert_approx_eq;

    use super::*;

    #[test]
    fn test_prediction_wrap_kernel_3() {
        let pdf = {
            let mut pdf = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.5, 0.5, 0.5, 0.5];

        let result = predict(&pdf, -1, &kernel, EdgeHandling::Wrap);
        let reference = [0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_extend_kernel_4() {
        let pdf = {
            let mut pdf = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.5, 0.5, 0.5, 0.5];

        let result = predict(&pdf, -1, &kernel, EdgeHandling::Constant(99.0));
        let reference = [
            4.95000000e+01,
            6.52189307e-18,
            -8.16487636e-19,
            1.78559758e-18,
            4.95000000e+01,
            9.90000000e+01,
            1.48500000e+02,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_wrap_kernel_4() {
        let pdf = {
            let mut pdf = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 8.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.25, 0.5, 0.125, 0.125];

        let result = predict(&pdf, 3, &kernel, EdgeHandling::Wrap);
        let reference = [
            0.29487179, 0.17948718, 0.08333333, 0.05128205, 0.05448718, 0.11217949, 0.22435897,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_wrap_kernel_5() {
        let pdf = {
            let mut pdf = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 8.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.25, 0.5, 0.125, 0.125, 10.0];

        let result = predict(&pdf, 3, &kernel, EdgeHandling::Wrap);
        let reference = [
            0.80769231, 1.20512821, 2.13461538, 4.15384615, 2.10576923, 0.11217949, 0.48076923,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }

    #[test]
    fn test_prediction_constant_kernel_4() {
        let pdf = {
            let mut pdf = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 8.0];
            normalize(&mut pdf);
            pdf
        };

        let kernel = [0.25, 0.5, 0.125, 0.125];

        let result = predict(&pdf, 3, &kernel, EdgeHandling::Constant(10.0));
        let reference = [
            10.0, 7.5, 2.50641026, 1.27564103, 0.05448718, 2.56089744, 7.51923077,
        ];
        dbg!(&result);
        dbg!(&reference);

        debug_assert_eq!(reference.len(), result.len());
        for i in 0..reference.len() {
            assert_approx_eq!(reference[i], result[i]);
        }
    }
}
