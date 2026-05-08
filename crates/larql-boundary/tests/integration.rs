//! Integration tests: full Phase 1 → 2 → 3 pipeline.

use larql_boundary::{
    codec::{bf16, int8},
    frame::{
        BoundaryAgreement, BoundaryCompression, BoundaryContract, BoundaryFrame, FallbackPolicy,
    },
    gate::{apply, BoundaryDecision, BoundaryGateConfig},
    metadata::compute,
};

const D: usize = 2560;

fn synthetic_residual() -> Vec<f32> {
    let mut r: Vec<f32> = (0..D).map(|i| (i as f32 * 0.01).sin() * 80.0).collect();
    r[100] = 94_208.0; // outlier
    r
}

fn peaked_logits(top: usize, n: usize) -> Vec<f32> {
    let mut l = vec![0.0f32; n];
    l[top] = 10.0;
    if top > 0 {
        l[top - 1] = 3.0;
    }
    l
}

// ── Phase 1 ──────────────────────────────────────────────────────────────

#[test]
fn bf16_roundtrip_no_overflow() {
    let r = synthetic_residual();
    let enc = bf16::encode(&r);
    let dec = bf16::decode(&enc);
    assert_eq!(dec.len(), r.len());
    for v in &dec {
        assert!(v.is_finite());
    }
}

#[test]
fn int8_roundtrip_correct_length() {
    let r = synthetic_residual();
    let payload = int8::encode(&r);
    let bytes = payload.to_bytes();
    assert_eq!(bytes.len(), 4 + D);
    let recovered = int8::decode(&int8::Payload::from_bytes(&bytes));
    assert_eq!(recovered.len(), D);
}

// ── Phase 2 ──────────────────────────────────────────────────────────────

#[test]
fn metadata_agrees_for_same_logits() {
    let logits = peaked_logits(42, 10_000);
    let meta = compute(&logits, Some(&logits));
    assert!(matches!(meta.boundary_agreement, BoundaryAgreement::Agrees));
    assert!(!meta.codec_fragile);
    assert!(meta.raw_log_prob_margin > 0.0);
}

#[test]
fn metadata_disagrees_for_different_top1() {
    let raw = peaked_logits(42, 10_000);
    let hat = peaked_logits(99, 10_000);
    let meta = compute(&raw, Some(&hat));
    assert!(matches!(
        meta.boundary_agreement,
        BoundaryAgreement::Disagrees
    ));
    assert!(meta.codec_fragile);
}

// ── Phase 3 ──────────────────────────────────────────────────────────────

#[test]
fn gate_compresses_confident_boundary() {
    let logits = peaked_logits(42, 10_000);
    let mut meta = compute(&logits, Some(&logits));
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 1.0,
        min_top1_prob: 0.5,
        ..Default::default()
    };
    let decision = apply(&mut meta, &config);
    assert!(
        matches!(decision, BoundaryDecision::CompressedOk { .. }),
        "expected CompressedOk, got {decision:?}"
    );
    assert!(!meta.boundary_fragile);
}

#[test]
fn gate_bf16_fallback_in_calibration_mode() {
    let logits = peaked_logits(42, 10_000);
    let mut meta = compute(&logits, Some(&logits));
    let config = BoundaryGateConfig::default(); // calibration_mode = true
    assert_eq!(apply(&mut meta, &config), BoundaryDecision::UseBf16);
}

#[test]
fn gate_rejects_codec_fragile() {
    let raw = peaked_logits(42, 10_000);
    let hat = peaked_logits(99, 10_000);
    let mut meta = compute(&raw, Some(&hat));
    let config = BoundaryGateConfig {
        calibration_mode: false,
        ..Default::default()
    };
    // codec_fragile → fallback
    assert_eq!(apply(&mut meta, &config), BoundaryDecision::UseBf16);
}

// ── Frame ─────────────────────────────────────────────────────────────────

#[test]
fn frame_continuation_safety() {
    let mut f = BoundaryFrame {
        version: 1,
        model_id: "m".into(),
        model_revision: "r".into(),
        tokenizer_revision: "t".into(),
        architecture: "a".into(),
        boundary_id: "b".into(),
        sequence_id: "s".into(),
        token_start: 0,
        token_end: 512,
        layer: 33,
        hidden_size: 2560,
        compression_scheme: BoundaryCompression::Int8Clip3Sigma,
        contract_level: BoundaryContract::ArgmaxNearEquivalentHighMargin,
        payload: vec![0u8; 2564],
        raw_top1_token: 42,
        raw_logit_margin: 5.0,
        raw_top1_prob: Some(0.9),
        compressed_top1_token: Some(42),
        boundary_agreement: BoundaryAgreement::Agrees,
        codec_fragile: false,
        boundary_fragile: false,
        fallback_policy: FallbackPolicy::Bf16Boundary,
        fallback_ref: None,
        calibration_run_id: None,
        residual_hash: None,
        token_hash: None,
    };

    assert!(f.is_safe_for_continuation());
    assert!(f.is_safe_for_routing());

    f.contract_level = BoundaryContract::Calibrating;
    assert!(!f.is_safe_for_continuation());
    assert!(!f.is_safe_for_routing());

    f.contract_level = BoundaryContract::ArgmaxNearEquivalentLowMargin;
    assert!(!f.is_safe_for_continuation());
    assert!(f.is_safe_for_routing());
}

// ── Full pipeline ─────────────────────────────────────────────────────────

#[test]
fn full_pipeline_encode_metadata_gate() {
    let residual = synthetic_residual();

    // Phase 1: encode
    let payload = int8::encode(&residual);
    let hat_residual = int8::decode(&payload);

    // Phase 2: compute metadata from logits (caller provides logits; we fake them here)
    let vocab = 100;
    let raw_logits: Vec<f32> = {
        let mut l = vec![0.0f32; vocab];
        l[42] = 8.0;
        l[17] = 2.0;
        l
    };
    let hat_logits: Vec<f32> = {
        // Use hat_residual's "logits" — simulate lm_head result: same top-1 if compression preserved
        let mut l = raw_logits.clone();
        // Small perturbation that doesn't change argmax
        l[17] = 2.1;
        l
    };
    let mut meta = compute(&raw_logits, Some(&hat_logits));
    assert!(matches!(meta.boundary_agreement, BoundaryAgreement::Agrees));

    // Phase 3: gate
    let config = BoundaryGateConfig {
        calibration_mode: false,
        min_log_prob_margin: 1.0,
        min_top1_prob: 0.3,
        ..Default::default()
    };
    let decision = apply(&mut meta, &config);
    assert!(matches!(decision, BoundaryDecision::CompressedOk { .. }));

    // Residual decoded correctly
    assert_eq!(hat_residual.len(), residual.len());
}
