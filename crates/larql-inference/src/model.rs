//! Model loading — re-exports from larql-vindex (canonical implementation).

// Re-export ModelWeights from larql-models
pub use larql_models::ModelWeights;

// Re-export loading functions from larql-vindex
pub use larql_vindex::{load_model_dir, resolve_model_path};
