//! Caller-side helpers: thin wrappers around `MatMul` that pick the
//! right method based on `Option<&dyn ComputeBackend>` (i.e. let
//! callers fall back to a CPU `ndarray` dot when no backend is
//! available).

use ndarray::Array2;

use super::ComputeBackend;

/// `dot_proj` through a backend: `a @ b^T`.
/// If `backend` is `None`, falls back to ndarray BLAS (CPU).
pub fn dot_proj_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul_transb(a.view(), b.view()),
        None => a.dot(&b.t()),
    }
}

/// `matmul` through a backend: `a @ b` (no transpose).
pub fn matmul_gpu(
    a: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    b: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    backend: Option<&dyn ComputeBackend>,
) -> Array2<f32> {
    match backend {
        Some(be) => be.matmul(a.view(), b.view()),
        None => a.dot(b),
    }
}
