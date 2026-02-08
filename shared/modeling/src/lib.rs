mod attention;
mod auto_config;
mod auto_model;
mod auto_tokenizer;
mod batcher;
mod causal_language_model;
mod device_utils;
mod distro;
mod dummy;
mod fp32_gradient_accumulator;
pub mod kernels;
pub mod matformer_c2;
pub mod matformer_helper;
pub mod metrics;
mod models;
pub mod muon;
mod optimizer;
mod parallelism;
#[cfg(feature = "python")]
mod python_causal_lm;
#[cfg(feature = "python")]
mod python_distributed_causal_lm;
#[cfg(feature = "python")]
mod python_distributed_trainer;
mod rms_norm;
mod rope;
mod safetensor_utils;
mod sampling;
mod token_output_stream;
mod trainer;
mod variable;

pub use attention::CausalSelfAttention;
pub use auto_config::{
    AttentionImplementation, AutoConfig, ModelConfig, ModelLoadError, PretrainedSource,
};
pub use auto_model::auto_model_for_causal_lm_from_pretrained;
pub use auto_tokenizer::{auto_tokenizer, AutoTokenizerError};
pub use batcher::Batcher;
pub use causal_language_model::{
    kd_loss_topk, CausalLM, CausalLanguageModel, EosToks, LanguageModelBuilder,
    LanguageModelConfig, LanguageModelForward, TeacherLogitTargets,
};
pub use device_utils::{get_optimal_devices, Devices};
pub use distro::{
    CompressDCT, Distro, DistroNormSidecar, DistroPeerMetadata, DistroResult, TransformDCT,
};
pub use dummy::{get_dummy_parameters, DummyModel};
pub use fp32_gradient_accumulator::Fp32GradientAccumulator;
pub use matformer_c2::{
    distillation_beta, DistillationCombineMode, DistillationConfig, MatformerStabilizationConfig,
    NormTierGainConfig, ResidualScaleConfig, SuffixGateConfig, SuffixGateSchedule,
};
pub use models::*;
pub use optimizer::{
    DistroAggregateMode, DistroApplyMode, DistroDilocoLiteConfig, DistroRawConfig,
    DistroRawMissingSidecarPolicy, DistroRawNormMode, DistroSignErrorFeedbackConfig,
    DistroTierProxConfig, DistroValueMode, Optimizer,
};
pub use parallelism::{
    unsharded_cpu_variables, AllReduce, ColumnParallelLinear, Communicator, CommunicatorId,
    CudaSynchronize, ParallelExpandHeads, ParallelismConfig, ReduceType, RowParallelLinear,
};
#[cfg(feature = "python")]
pub use python_causal_lm::{PythonCausalLM, PythonCausalLMError, PythonModelConfig};
#[cfg(feature = "python")]
pub use python_distributed_causal_lm::{
    PythonDistributedCausalLM, PythonDistributedCausalLMError, TorchDistributedCommunicator,
};
#[cfg(feature = "python")]
pub use python_distributed_trainer::{
    NopBarrier, PythonDistributedTrainer, PythonDistributedTrainerError,
};
pub use rms_norm::RMSNorm;
pub use rope::{default_rope, rotate_half, yarn_get_mscale, RoPECache, RoPEConfig, RoPEType};
pub use safetensor_utils::{
    load_safetensors_into_variables, save_tensors_into_safetensors, LoadSafetensorsError,
    SaveSafetensorsError,
};
pub use sampling::{LogitsProcessor, Sampling};
pub use token_output_stream::TokenOutputStream;
pub use trainer::{
    ApplyDistroResultError, Batch, BatchData, BatchDataCPU, BatchDataGPU, DataParallel,
    LocalTrainer, ParallelModels, TrainOutput, Trainer, TrainerThreadCommunicationError,
};
pub use variable::{StableVarStoreIterator, StableVariableIterator, Variable};

#[allow(unused)]
pub fn set_torch_rng_seed() {
    use rand::Rng;

    let seed: i64 = rand::rng().random();
    tch::manual_seed(seed);
    println!("torch seed set to: {seed}");
}

pub fn set_suggested_env_vars() {
    std::env::set_var("NCCL_P2P_DIRECT_DISABLE", "1");
    std::env::set_var("NCCL_LAUNCH_MODE", "GROUP");
}
