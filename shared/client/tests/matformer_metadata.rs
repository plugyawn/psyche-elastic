use psyche_client::matformer::{
    annotate_matformer_checkpoint_config, ensure_matformer_checkpoint_metadata,
    infer_matformer_checkpoint_metadata,
};
use serde_json::json;

#[test]
fn infer_metadata_from_explicit_tier() {
    let config = json!({
        "intermediate_size": 512,
        "matformer_tier": 1,
    });
    let metadata = infer_matformer_checkpoint_metadata(&config);
    assert_eq!(metadata.tier, Some(1));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
}

#[test]
fn infer_metadata_from_base_size() {
    let config = json!({
        "intermediate_size": 512,
        "matformer_base_intermediate_size": 1024,
    });
    let metadata = infer_matformer_checkpoint_metadata(&config);
    assert_eq!(metadata.tier, Some(1));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
}

#[test]
fn infer_metadata_without_matformer_fields() {
    let config = json!({
        "intermediate_size": 512,
    });
    let metadata = infer_matformer_checkpoint_metadata(&config);
    assert_eq!(metadata.tier, None);
    assert_eq!(metadata.base_intermediate_size, Some(512));
}

#[test]
fn infer_metadata_rejects_non_divisible_ratio() {
    let config = json!({
        "intermediate_size": 600,
        "matformer_base_intermediate_size": 1000,
    });
    let metadata = infer_matformer_checkpoint_metadata(&config);
    assert_eq!(metadata.tier, None);
    assert_eq!(metadata.base_intermediate_size, Some(1000));
}

#[test]
fn annotate_config_sets_tier_and_base_size() {
    let mut config = json!({
        "intermediate_size": 512,
    });
    let metadata = annotate_matformer_checkpoint_config(&mut config, Some(1024));
    assert_eq!(metadata.tier, Some(1));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
    assert_eq!(config["matformer_tier"], 1);
    assert_eq!(config["matformer_base_intermediate_size"], 1024);
    assert_eq!(config["intermediate_size"], 512);
}

#[test]
fn annotate_config_respects_existing_base_size() {
    let mut config = json!({
        "intermediate_size": 512,
        "matformer_base_intermediate_size": 1024,
    });
    let metadata = annotate_matformer_checkpoint_config(&mut config, None);
    assert_eq!(metadata.tier, Some(1));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
    assert_eq!(config["matformer_tier"], 1);
    assert_eq!(config["matformer_base_intermediate_size"], 1024);
    assert_eq!(config["intermediate_size"], 512);
}

#[test]
fn ensure_metadata_uses_manifest_base_size() {
    let mut config = json!({
        "intermediate_size": 512,
    });
    let metadata = ensure_matformer_checkpoint_metadata(&mut config, Some(1024), None);
    assert_eq!(metadata.tier, Some(1));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
    assert_eq!(config["matformer_tier"], 1);
    assert_eq!(config["matformer_base_intermediate_size"], 1024);
}

#[test]
fn ensure_metadata_uses_slice_tier_hint() {
    let mut config = json!({
        "intermediate_size": 256,
    });
    let metadata = ensure_matformer_checkpoint_metadata(&mut config, None, Some(2));
    assert_eq!(metadata.tier, Some(2));
    assert_eq!(metadata.base_intermediate_size, Some(1024));
    assert_eq!(config["matformer_tier"], 2);
    assert_eq!(config["matformer_base_intermediate_size"], 1024);
}

#[test]
fn ensure_metadata_respects_existing_fields() {
    let mut config = json!({
        "intermediate_size": 512,
        "matformer_tier": 2,
        "matformer_base_intermediate_size": 2048,
    });
    let metadata = ensure_matformer_checkpoint_metadata(&mut config, Some(1024), Some(1));
    assert_eq!(metadata.tier, Some(2));
    assert_eq!(metadata.base_intermediate_size, Some(2048));
    assert_eq!(config["matformer_tier"], 2);
    assert_eq!(config["matformer_base_intermediate_size"], 2048);
}
