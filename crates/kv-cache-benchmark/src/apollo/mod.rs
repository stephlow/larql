//! Apollo — re-exported from `larql_inference::engines::apollo`.
//!
//! The implementation now lives in larql-inference. This module re-exports
//! all public types so existing benchmark code continues to compile unchanged.

pub use larql_inference::engines::apollo::routing::RoutingQuery;
pub use larql_inference::engines::apollo::store::{ApolloStore, StoreManifest};
pub use larql_inference::engines::apollo::{
    ApolloEngine, ApolloError, InjectionConfig, QueryTrace, RoutingIndex, VecInjectEntry,
};

// Sub-modules re-exported in case tests import from them directly.
pub use larql_inference::engines::apollo::entry;
pub use larql_inference::engines::apollo::routing;
pub use larql_inference::engines::apollo::store;
