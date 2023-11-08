/*!
This module implements the linear Kalman filter
*/

use nalgebra::allocator::Allocator;
use nalgebra::base::dimension::DimName;
use nalgebra::{DefaultAllocator, OMatrix, RealField, OVector};
use num_traits::Float;

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
#[derive(Debug)]
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
    ///Posterior (updated) state covariance matrix.
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

        if B.is_some() && u.is_some() {
            self.x = F * self.x.clone() + B.unwrap() * u.unwrap();
        } else {
            self.x = F * self.x.clone();
        }

        self.P = ((F * self.P.clone()) * F.transpose()) * self.alpha_sq + Q;

        self.x_prior = self.x.clone();
        self.P_prior = self.P.clone();
    }

    /// Add a new measurement (z) to the Kalman filter.
    pub fn update(
        &mut self,
        z: &OVector<F, DimZ>,
        R: Option<&OMatrix<F, DimZ, DimZ>>,
        H: Option<&OMatrix<F, DimZ, DimX>>,
    ) {
        let R = R.unwrap_or(&self.R);
        let H = H.unwrap_or(&self.H);

        self.y = z - H * &self.x;

        let PHT = self.P.clone() * H.transpose();
        self.S = H * &PHT + R;

        self.SI = self.S.clone().try_inverse().unwrap();

        self.K = PHT * &self.SI;

        self.x = &self.x + &self.K * &self.y;

        let I_KH = OMatrix::<F, DimX, DimX>::identity() - &self.K * H;
        self.P =
            ((I_KH.clone() * &self.P) * I_KH.transpose()) + ((&self.K * R) * &self.K.transpose());

        self.z = Some(z.clone());
        self.x_post = self.x.clone();
        self.P_post = self.P.clone();
    }

    /// Predict state (prior) using the Kalman filter state propagation equations.
    /// Only x is updated, P is left unchanged.
    pub fn predict_steadystate(
        &mut self,
        u: Option<&OVector<F, DimU>>,
        B: Option<&OMatrix<F, DimX, DimU>>,
    ) {
        let B = if B.is_some() { B } else { self.B.as_ref() };

        if B.is_some() && u.is_some() {
            self.x = &self.F * &self.x + B.unwrap() * u.unwrap();
        } else {
            self.x = &self.F * &self.x;
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
            if B.is_some() && u.is_some() {
                F * &self.x + B.unwrap() * u.unwrap()
            } else {
                F * &self.x
            }
        };

        let P = ((F * P) * FT) * self.alpha_sq + Q;

        (x, P)
    }

    ///  Computes the new estimate based on measurement `z` and returns it without altering the state of the filter.
    pub fn get_update(&self, z: &OVector<F, DimZ>) -> (OVector<F, DimX>, OMatrix<F, DimX, DimX>) {
        let R = &self.R;
        let H = &self.H;
        let P = &self.P;
        let x = &self.x;

        let y = z - H * &self.x;

        let PHT = &(P * H.transpose());

        let S = H * PHT + R;
        let SI = S.try_inverse().unwrap();

        let K = &(PHT * SI);

        let x = x + K * y;

        let I_KH = &(OMatrix::<F, DimX, DimX>::identity() - (K * H));

        let P = ((I_KH * P) * I_KH.transpose()) + ((K * R) * &K.transpose());

        (x, P)
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

#[cfg(test)]
mod tests {
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
            kf.update(&z, None, None);
            assert_approx_eq!(zf, kf.z.clone().unwrap()[0]);
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
            kf.update(&z, None, None);
            kf.predict(None, None, None, None);
            // This matches the results from an equivalent filterpy filter.
            assert_approx_eq!(
                kf.x[0],
                if t == 0 { 0.0099502487 } else { t as f64 + 1.0 },
                0.05
            );
        }
    }
}
