//! HTTP integration tests: SessionManager tests.

mod common;
use common::*;

use larql_server::session::SessionManager;

// ══════════════════════════════════════════════════════════════
// ASYNC STATE / SESSION MANAGER TESTS
// ══════════════════════════════════════════════════════════════

#[tokio::test]
async fn session_manager_list_empty_for_unknown_session() {
    let sm = SessionManager::new(3600);
    let patches = sm.list_patches("session-xyz").await;
    assert!(patches.is_empty());
}

#[tokio::test]
async fn session_manager_apply_patch_and_list() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    // Pre-create the session with get_or_create (uses read().await, safe in async).
    // apply_patch's or_insert_with calls blocking_read only when the session doesn't
    // exist, so we must create it first.
    sm.get_or_create("sess-1", &m).await;

    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };

    let (op_count, active) = sm.apply_patch("sess-1", &m, patch).await;
    assert_eq!(op_count, 1);
    assert_eq!(active, 1);

    let list = sm.list_patches("sess-1").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "my-patch");
}

#[tokio::test]
async fn session_manager_remove_nonexistent_patch_returns_err() {
    let sm = SessionManager::new(3600);
    let m = model("test");
    // Pre-create the session, then apply one patch.
    sm.get_or_create("sess-1", &m).await;
    let patch = larql_vindex::VindexPatch {
        version: 1,
        base_model: "test".into(),
        base_checksum: None,
        created_at: "2026-04-26".into(),
        description: Some("my-patch".into()),
        author: None,
        tags: vec![],
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    };
    sm.apply_patch("sess-1", &m, patch).await;

    let err = sm.remove_patch("sess-1", "nonexistent").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}

#[tokio::test]
async fn session_manager_remove_patch_by_name() {
    let sm = SessionManager::new(3600);
    let m = model("test");

    // Pre-create session, then apply two patches.
    sm.get_or_create("sess-2", &m).await;
    for name in &["patch-a", "patch-b"] {
        let patch = larql_vindex::VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-04-26".into(),
            description: Some((*name).into()),
            author: None,
            tags: vec![],
            operations: vec![larql_vindex::PatchOp::Delete {
                layer: 0,
                feature: 1,
                reason: None,
            }],
        };
        sm.apply_patch("sess-2", &m, patch).await;
    }

    let remaining = sm.remove_patch("sess-2", "patch-a").await.unwrap();
    assert_eq!(remaining, 1);

    let list = sm.list_patches("sess-2").await;
    assert_eq!(list.len(), 1);
    assert_eq!(list[0]["name"], "patch-b");
}

#[tokio::test]
async fn session_manager_remove_from_unknown_session_returns_err() {
    let sm = SessionManager::new(3600);
    let err = sm.remove_patch("no-such-session", "any-patch").await;
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("not found"));
}
