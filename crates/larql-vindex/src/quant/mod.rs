//! FP4/FP8 build-time operations on a vindex.
//!
//! - `scan`:    Q1 compliance measurement — read-only, no output
//!              side effects. Used by `convert` as a self-policing
//!              gate and by the `fp4_q1_scan` example binary.
//! - `convert`: `vindex_to_fp4` — reads an existing vindex, writes
//!              a new FP4/FP8 vindex per the chosen policy. Used by
//!              the `fp4_convert` example binary and the
//!              `larql convert quantize fp4` CLI subcommand.
//!
//! Runtime FP4 data structures (the `Fp4Storage` attached to a
//! loaded `VectorIndex`) live elsewhere — see
//! `crate::index::fp4_storage` and `crate::format::fp4_storage`.

pub mod scan;
pub mod convert;
pub mod convert_q4k;

pub use scan::{
    scan_projection, scan_vindex, BucketQuantiles, ComplianceThreshold,
    Dtype, GranularityStats, LayerStats, ProjectionReport, ScanConfig,
    VindexComplianceReport, PROJECTIONS,
};
pub use convert::{
    vindex_to_fp4, Fp4ConvertConfig, Fp4ConvertReport, Policy,
    ProjectionAction, ProjectionOutcome,
};
pub use convert_q4k::{
    vindex_to_q4k, Q4kConvertConfig, Q4kConvertReport,
};
