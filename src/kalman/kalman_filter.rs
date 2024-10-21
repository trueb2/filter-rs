/*!
This module implements the linear Kalman filter
*/

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{DefaultAllocator, OMatrix, OVector, RealField};
use num_traits::Float;

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Implements a Kalman filter.
/// For a detailed explanation, see the excellent book Kalman and Bayesian
/// Filters in Python [1]_. The book applies also for this Rust implementation and all functions
/// should works similar with minor changes due to language differences.
///
///  References
///    ----------
///
///    .. [1] Roger Labbe. "Kalman and Bayesian Filters in Python"
///       https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
///
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct KalmanFilter<F, DimX, DimZ, DimU>
where
    F: RealField + Float,
    DimX: DimName,
    DimZ: DimName,
    DimU: DimName,
    DefaultAllocator: Allocator<F, DimX>
        + Allocator<F, DimZ>
        + Allocator<F, DimX, DimZ>
        + Allocator<F, DimZ, DimX>
        + Allocator<F, DimZ, DimZ>
        + Allocator<F, DimX, DimX>
        + Allocator<F, DimU>
        + Allocator<F, DimX, DimU>,
{
    /// Current state estimate.
    pub x: OVector<F, DimX>,
    /// Current state covariance matrix.
    pub P: OMatrix<F, DimX, DimX>,
    /// Prior (predicted) state estimate.
    pub x_prior: OVector<F, DimX>,
    /// Prior (predicted) state covariance matrix.
    pub P_prior: OMatrix<F, DimX, DimX>,
    /// Posterior (updated) state estimate.
    pub x_post: OVector<F, DimX>,
    /// Posterior (updated) state covariance matrix.
    pub P_post: OMatrix<F, DimX, DimX>,
    /// Last measurement
    pub z: Option<OVector<F, DimZ>>,
    /// Measurement noise matrix.
    pub R: OMatrix<F, DimZ, DimZ>,
    /// OMatrix<F, DimZ, DimZ>,
    pub Q: OMatrix<F, DimX, DimX>,
    /// Control transition matrix
    pub B: Option<OMatrix<F, DimX, DimU>>,
    /// State Transition matrix.
    pub F: OMatrix<F, DimX, DimX>,
    /// Measurement function.
    pub H: OMatrix<F, DimZ, DimX>,
    /// Residual of the update step.
    pub y: OVector<F, DimZ>,
    /// Kalman gain of the update step.
    pub K: OMatrix<F, DimX, DimZ>,
    /// System uncertainty (P projected to measurement space).
    pub S: OMatrix<F, DimZ, DimZ>,
    /// Inverse system uncertainty.
    pub SI: OMatrix<F, DimZ, DimZ>,
    /// Fading memory setting.
    pub alpha_sq: F,
}

/// Kalman filtering may error in some cases.
#[derive(Debug, Clone, Copy)]
pub enum KalmanError {
    /// The matrix is not invertible.
    NotInvertible,
    /// The length of the input arrays are not equal.
    LengthMismatch,
    /// Invalid length of the input arrays.
    InvalidLength,
}

#[allow(non_snake_case)]
impl<F, DimX, DimZ, DimU> KalmanFilter<F, DimX, DimZ, DimU>
where
    F: RealField + Float,
    DimX: DimName,
    DimZ: DimName,
    DimU: DimName,
    DefaultAllocator: Allocator<F, DimX>
        + Allocator<F, DimZ>
        + Allocator<F, DimX, DimZ>
        + Allocator<F, DimZ, DimX>
        + Allocator<F, DimZ, DimZ>
        + Allocator<F, DimX, DimX>
        + Allocator<F, DimU>
        + Allocator<F, DimX, DimU>,
{
    /// Predict next state (prior) using the Kalman filter state propagation equations.
    pub fn predict(
        &mut self,
        u: Option<&OVector<F, DimU>>,
        B: Option<&OMatrix<F, DimX, DimU>>,
        F: Option<&OMatrix<F, DimX, DimX>>,
        Q: Option<&OMatrix<F, DimX, DimX>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };
        let F = F.unwrap_or(&self.F);
        let Q = Q.unwrap_or(&self.Q);

        match (B, u) {
            (Some(B), Some(u)) => self.x = F * &self.x + B * u,
            _ => self.x = F * &self.x,
        }

        self.P = ((F * &self.P) * F.transpose()) * self.alpha_sq + Q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(
        &mut self,
        z: &OVector<F, DimZ>,
        R: Option<&OMatrix<F, DimZ, DimZ>>,
        H: Option<&OMatrix<F, DimZ, DimX>>,
    ) -> Result<(), KalmanError> {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H * &self.x;

        let PHT = &self.P * H.transpose();
        self.S = H * &PHT + R;

        self.SI = self
            .S
            .clone()
            .try_inverse()
            .ok_or(KalmanError::NotInvertible)?;

        self.K = PHT * &self.SI;

        self.x = &self.x + &self.K * &self.y;

        let I_KH = OMatrix::<F, DimX, DimX>::identity() - &self.K * H;
        self.P = ((&I_KH * &self.P) * I_KH.transpose()) + ((&self.K * R) * &self.K.transpose());

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();

        Ok(())
    }

    /// Compute the Rauch-Tung-Striebel Kalman smoother on a set of means and covariances from the Kalman filter.
    ///
    /// # Arguments
    /// Xs: array of the means (states) of the Kalman filter.
    /// Ps: array of the covariances of the Kalman filter.
    /// Fs: Optional array of the state transition matrices of the Kalman filter.
    /// Qs: Optional array of the process noise matrices of the Kalman filter.
    /// result: Optional mutable reference to a `RTSSmoothedResults` struct to store the results.
    ///     This allows reusing an existing struct instead of allocating a new one.
    ///
    /// # Returns
    /// A smoothed state, covariance, smoother gain, and predicted covariance.
    #[cfg(feature = "alloc")]
    pub fn rts_smoother(
        &mut self,
        Xs: &[OVector<F, DimX>],
        Ps: &[OMatrix<F, DimX, DimX>],
        Fs: Option<&[OMatrix<F, DimX, DimX>]>,
        Qs: Option<&[OMatrix<F, DimX, DimX>]>,
        mut result: RTSSmoothedResults<F, DimX>,
    ) -> Result<RTSSmoothedResults<F, DimX>, KalmanError> {
        if Xs.len() != Ps.len() {
            return Err(KalmanError::LengthMismatch);
        } else if Xs.len() < 2 {
            return Err(KalmanError::InvalidLength);
        }

        let n = Xs.len();

        let mut fsv = None;
        let mut qsv = None;
        if Fs.is_none() {
            fsv = Some(vec![self.F.clone(); n]);
        }
        if Qs.is_none() {
            qsv = Some(vec![self.Q.clone(); n]);
        }
        let Fs = Fs.unwrap_or(fsv.as_ref().unwrap());
        let Qs = Qs.unwrap_or(qsv.as_ref().unwrap());

        // Re-use the buffers in the result struct
        result.clear();
        result.K.resize(n, OMatrix::<F, DimX, DimX>::zeros());
        result.x.reserve(Xs.len());
        result.P.reserve(Ps.len());
        result.Pp.reserve(Ps.len());
        result.x.extend_from_slice(Xs);
        result.P.extend_from_slice(Ps);
        result.Pp.extend_from_slice(Ps);

        let x = &mut result.x;
        let K = &mut result.K;
        let P = &mut result.P;
        let Pp = &mut result.Pp;

        for k in (0..n - 1).rev() {
            Pp[k] = (&Fs[k + 1] * &P[k]) * Fs[k + 1].transpose() + &Qs[k + 1];
            K[k] = (&P[k] * Fs[k + 1].transpose())
                * Pp[k]
                    .clone()
                    .try_inverse()
                    .ok_or(KalmanError::NotInvertible)?;
            let xk = &K[k] * (&x[k + 1] - &Fs[k + 1] * &x[k]);
            x[k] += xk;
            let pk = &K[k] * (&P[k + 1] - &Pp[k]) * &K[k].transpose();
            P[k] += pk;
        }

        Ok(result)
    }

    /// Predict state (prior) using the Kalman filter state propagation equations.
    /// Only x is updated, P is left unchanged.
    pub fn predict_steadystate(
        &mut self,
        u: Option<&OVector<F, DimU>>,
        B: Option<&OMatrix<F, DimX, DimU>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        match (B, u) {
            (Some(B), Some(u)) => self.x = &self.F * &self.x + B * u,
            _ => self.x = &self.F * &self.x,
        }

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter without recomputing the Kalman gain K,
    /// the state covariance P, or the system uncertainty S.
    pub fn update_steadystate(&mut self, z: &OVector<F, DimZ>) {
        self.y = z - &self.H * &self.x;
        self.x = &self.x + &self.K * &self.y;

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    /// Predicts the next state of the filter and returns it without altering the state of the filter.
    pub fn get_prediction(
        &self,
        u: Option<&OVector<F, DimU>>,
    ) -> (OVector<F, DimX>, OMatrix<F, DimX, DimX>) {
        let Q = &self.Q;
        let F = &self.F;
        let P = &self.P;
        let FT = F.transpose();

        let B = self.B.as_ref();
        let x = {
            match (B, u) {
                (Some(B), Some(u)) => F * &self.x + B * u,
                _ => F * &self.x,
            }
        };

        let P = ((F * P) * FT) * self.alpha_sq + Q;

        (x, P)
    }

    ///  Computes the new estimate based on measurement `z` and returns it without altering the state of the filter.
    #[allow(clippy::type_complexity)]
    pub fn get_update(
        &self,
        z: &OVector<F, DimZ>,
    ) -> Result<(OVector<F, DimX>, OMatrix<F, DimX, DimX>), KalmanError> {
        let R = &self.R;
        let H = &self.H;
        let P = &self.P;
        let x = &self.x;

        let y = z - H * &self.x;

        let PHT = &(P * H.transpose());

        let S = H * PHT + R;
        let SI = S.try_inverse().ok_or(KalmanError::NotInvertible)?;

        let K = &(PHT * SI);

        let x = x + K * y;

        let I_KH = &(OMatrix::<F, DimX, DimX>::identity() - (K * H));

        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        Ok((x, P))
    }

    /// Returns the residual for the given measurement (z). Does not alter the state of the filter.
    pub fn residual_of(&self, z: &OVector<F, DimZ>) -> OVector<F, DimZ> {
        z - (&self.H * &self.x_prior)
    }

    /// Helper function that converts a state into a measurement.
    pub fn measurement_of_state(&self, x: &OVector<F, DimX>) -> OVector<F, DimZ> {
        &self.H * x
    }
}

#[allow(non_snake_case)]
impl<F, DimX, DimZ, DimU> Default for KalmanFilter<F, DimX, DimZ, DimU>
where
    F: RealField + Float,
    DimX: DimName,
    DimZ: DimName,
    DimU: DimName,
    DefaultAllocator: Allocator<F, DimX>
        + Allocator<F, DimZ>
        + Allocator<F, DimX, DimZ>
        + Allocator<F, DimZ, DimX>
        + Allocator<F, DimZ, DimZ>
        + Allocator<F, DimX, DimX>
        + Allocator<F, DimU>
        + Allocator<F, DimX, DimU>,
{
    /// Returns a Kalman filter initialised with default parameters.
    fn default() -> Self {
        let x = OVector::<F, DimX>::from_element(F::one());
        let P = OMatrix::<F, DimX, DimX>::identity();
        let Q = OMatrix::<F, DimX, DimX>::identity();
        let F = OMatrix::<F, DimX, DimX>::identity();
        let H = OMatrix::<F, DimZ, DimX>::from_element(F::zero());
        let R = OMatrix::<F, DimZ, DimZ>::identity();
        let alpha_sq = F::one();

        let z = None;

        let K = OMatrix::<F, DimX, DimZ>::from_element(F::zero());
        let y = OVector::<F, DimZ>::from_element(F::one());
        let S = OMatrix::<F, DimZ, DimZ>::from_element(F::zero());
        let SI = OMatrix::<F, DimZ, DimZ>::from_element(F::zero());

        let x_prior = x.clone();
        let P_prior = P.clone();

        let x_post = x.clone();
        let P_post = P.clone();

        KalmanFilter {
            x,
            P,
            x_prior,
            P_prior,
            x_post,
            P_post,
            z,
            R,
            Q,
            B: None,
            F,
            H,
            y,
            K,
            S,
            SI,
            alpha_sq,
        }
    }
}

/// Results from the Rauch-Tung-Striebel smoother.
#[cfg(feature = "alloc")]
#[allow(non_snake_case)]
#[derive(Default, Debug, Clone)]
pub struct RTSSmoothedResults<F, DimX>
where
    F: RealField + Float,
    DimX: DimName,
    DefaultAllocator: Allocator<F, DimX> + Allocator<F, DimX, DimX>,
{
    /// Smoothed state means.
    pub x: Vec<OVector<F, DimX>>,
    /// Smoothed state covariances.
    pub P: Vec<OMatrix<F, DimX, DimX>>,
    /// Smoother gain per step.
    pub K: Vec<OMatrix<F, DimX, DimX>>,
    /// Predicted state covariances.
    pub Pp: Vec<OMatrix<F, DimX, DimX>>,
}

impl<F, DimX> RTSSmoothedResults<F, DimX>
where
    F: RealField + Float,
    DimX: DimName,
    DefaultAllocator: Allocator<F, DimX> + Allocator<F, DimX, DimX>,
{
    /// Create a new `RTSSmoothedResults` struct with empty buffers.
    /// Some dimensions may not have a Default impl
    pub fn new() -> Self {
        RTSSmoothedResults {
            x: vec![],
            P: vec![],
            K: vec![],
            Pp: vec![],
        }
    }

    /// Clear the buffers without deallocating them.
    pub fn clear(&mut self) {
        self.x.clear();
        self.P.clear();
        self.K.clear();
        self.Pp.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::dbg;

    use assert_approx_eq::assert_approx_eq;
    use nalgebra::base::Vector1;
    use nalgebra::{Matrix1, Matrix2, Vector2, U1, U2};

    use super::*;

    #[test]
    fn test_univariate_kf_setup() {
        let mut kf: KalmanFilter<f32, U1, U1, U1> = KalmanFilter::<f32, U1, U1, U1>::default();

        for i in 0..1000 {
            let zf = i as f32;
            let z = Vector1::new(zf);
            kf.predict(None, None, None, None);
            kf.update(&z, None, None).unwrap();
            assert_approx_eq!(zf, kf.z.unwrap()[0]);
        }
    }

    #[test]
    fn test_1d_reference() {
        let mut kf: KalmanFilter<f64, U2, U1, U1> = KalmanFilter::default();

        kf.x = Vector2::new(2.0, 0.0);
        kf.F = Matrix2::new(1.0, 1.0, 0.0, 1.0);
        kf.H = Vector2::new(1.0, 0.0).transpose();
        kf.P *= 1000.0;
        kf.R = Matrix1::new(5.0);
        kf.Q = Matrix2::repeat(0.0001);

        for t in 0..100 {
            let z = Vector1::new(t as f64);
            kf.update(&z, None, None).unwrap();
            kf.predict(None, None, None, None);
            // This matches the results from an equivalent filterpy filter.
            assert_approx_eq!(
                kf.x[0],
                if t == 0 { 0.0099502487 } else { t as f64 + 1.0 },
                0.05
            );
        }
    }

    #[allow(non_snake_case)]
    #[cfg(feature = "alloc")]
    #[test]
    fn test_rts_smoother() {
        let mut kf: KalmanFilter<f64, U2, U1, U1> = KalmanFilter::default();
        kf.x = Vector2::new(2.0, 0.0);
        kf.F = Matrix2::new(1.0, 1.0, 0.0, 1.0);
        kf.H = Vector2::new(1.0, 0.0).transpose();
        kf.P *= 1000.0;
        kf.R = Matrix1::new(5.0);
        kf.Q = Matrix2::repeat(0.0001);

        let mut xs = vec![];
        let mut Ps = vec![];
        for t in 0..100 {
            let z = Vector1::new(t as f64);
            kf.update(&z, None, None).unwrap();
            kf.predict(None, None, None, None);
            xs.push(kf.x);
            Ps.push(kf.P);
            // This matches the results from an equivalent filterpy filter.
            assert_approx_eq!(
                kf.x[0],
                if t == 0 { 0.0099502487 } else { t as f64 + 1.0 },
                0.05
            );
        }

        let rts = kf
            .rts_smoother(&xs, &Ps, None, None, RTSSmoothedResults::new())
            .unwrap();
        dbg!(&rts.x);
        dbg!(&rts.P);
        dbg!(&rts.K);
        dbg!(&rts.Pp);
    }
}
