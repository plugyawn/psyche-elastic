use std::sync::atomic::{AtomicI64, Ordering};

/// Global override for MatFormer FFN intermediate width.
///
/// This is intentionally a process-global knob so the trainer can temporarily
/// run a "core subnet" forward/backward on tier-0 without constructing a second
/// model instance. Models that support MatFormer consult this during forward.
static OVERRIDE_INTERMEDIATE_SIZE: AtomicI64 = AtomicI64::new(0);

pub fn set_override_intermediate_size(size: Option<i64>) {
    OVERRIDE_INTERMEDIATE_SIZE.store(size.unwrap_or(0), Ordering::Relaxed);
}

pub fn get_override_intermediate_size() -> Option<i64> {
    let v = OVERRIDE_INTERMEDIATE_SIZE.load(Ordering::Relaxed);
    (v > 0).then_some(v)
}

