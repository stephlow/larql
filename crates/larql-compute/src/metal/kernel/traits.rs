//! `TiledKernel` — marker trait that lets a shader module own its own
//! kernel name + dispatch geometry as compile-time constants.
//!
//! The shader source already lives in `shaders/<name>.rs`. Adding a
//! `pub struct Kernel; impl TiledKernel for Kernel { … }` block to
//! that file co-locates name + geometry + source. Binding the
//! pipeline becomes a one-line call to
//! [`KernelHandle::from_kernel::<…::Kernel>(device, library)`](super::KernelHandle::from_kernel).
//! Bumping a shader (e.g. `q4_matvec_v4` → `_v6`) = change the type
//! parameter at the binding site. No magic strings at the binding
//! site, no chance of geometry drifting from the kernel.

/// A simdgroup-tiled compute kernel that needs `dispatch_thread_groups`
/// geometry to drive correctly. Implemented by a marker `Kernel` type
/// inside each tiled-shader module.
///
/// Flat-dispatch kernels (one thread per output element, driven by
/// `dispatch_threads`) don't need geometry and shouldn't implement
/// this trait — they're plain `ComputePipelineState`s.
pub trait TiledKernel {
    /// Metal kernel function name as it appears in
    /// `kernel void <name>(…)` in the shader source.
    const KERNEL_NAME: &'static str;
    /// Output rows the kernel covers per threadgroup.
    const ROWS_PER_TG: u64;
    /// Threads per threadgroup the kernel is sized for.
    const THREADS_PER_TG: u64;
}
