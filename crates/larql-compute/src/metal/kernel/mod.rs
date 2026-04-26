//! Pipeline + dispatch geometry handle, kernel-name registry, and
//! related helpers.
//!
//! ## Why this module exists
//!
//! Shaders with simdgroup-tiled row mapping (q4_matvec_v4, q4k_matvec,
//! q4k_ffn_gate_up, …) hardcode their per-TG row coverage. The
//! dispatch wrapper has to compute `num_tgs = num_rows.div_ceil
//! (rows_per_tg)` and request `threads_per_tg` threads in agreement
//! with the kernel's row map. Importing those constants from a
//! *different* shader module while the pipeline is built from the
//! kernel that's actually loaded is exactly how the q4_matvec_v4
//! 75 %-row-drop bug landed (closed 2026-04-25 — see ROADMAP.md ship
//! log).
//!
//! ## Layout
//!
//! - `traits`: [`TiledKernel`] — marker trait a shader module
//!   implements to expose its kernel name + dispatch geometry as
//!   compile-time constants. The shader source, name, and geometry
//!   then all live in the same file.
//! - `handle`: [`KernelHandle`] — pipeline state + geometry + name,
//!   bundled. Construction goes through
//!   [`KernelHandle::from_kernel::<K: TiledKernel>`](handle::KernelHandle::from_kernel),
//!   so binding sites read constants by *path*, not by hand-typed
//!   strings. Construction also asserts pipeline
//!   `maxTotalThreadsPerThreadgroup` ≥ requested `threads_per_tg`
//!   so silent simdgroup drop is caught at startup, not at
//!   goldens-fail time.

pub mod handle;
pub mod traits;

pub use handle::KernelHandle;
pub use traits::{get_shader_pipeline, ShaderKernel, TiledKernel};
