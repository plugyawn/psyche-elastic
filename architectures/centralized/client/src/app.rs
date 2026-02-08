use anyhow::{Error, Result};
use bytemuck::Zeroable;
use hf_hub::Repo;
use psyche_centralized_shared::{
    ClientCapabilities, ClientId, ClientToServerMessage, ServerToClientMessage, TrainingAssignment,
};
use psyche_client::{
    read_identity_secret_key, Client, ClientTUI, ClientTUIState, RunInitConfig, TrainArgs, NC,
};
use psyche_coordinator::{model, Coordinator, HealthChecks};
use psyche_metrics::ClientMetrics;
use psyche_network::{
    allowlist, AuthenticatableIdentity, EndpointId, NetworkTUIState, NetworkTui, SecretKey,
    TcpClient,
};
use psyche_tui::logging::LoggerWidget;
use psyche_tui::{CustomWidget, TabbedWidget};
use psyche_watcher::{
    Backend as WatcherBackend, CoordinatorTui, ModelSchemaInfo, OpportunisticData,
};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::Sender;
use tokio::time::interval;
use tokio::{select, sync::mpsc, time::Interval};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

pub(super) type Tabs = TabbedWidget<(ClientTUI, CoordinatorTui, NetworkTui, LoggerWidget)>;
pub const TAB_NAMES: [&str; 4] = ["Client", "Coordinator", "Network", "Logger"];
pub type TabsData = <Tabs as CustomWidget>::Data;

pub enum ToSend {
    Witness(Box<OpportunisticData>),
    HealthCheck(HealthChecks<ClientId>),
    Checkpoint(model::HubRepo),
    Schema(ModelSchemaInfo),
}

struct Backend {
    allowlist: allowlist::AllowDynamic,
    rx: mpsc::UnboundedReceiver<Coordinator<ClientId>>,
    tx: mpsc::UnboundedSender<ToSend>,
}

#[async_trait::async_trait]
impl WatcherBackend<ClientId> for Backend {
    async fn wait_for_new_state(&mut self) -> Result<Coordinator<ClientId>> {
        let new_state = self
            .rx
            .recv()
            .await
            .ok_or(Error::msg("watcher backend rx channel closed"))?;
        self.allowlist.set(
            new_state
                .epoch_state
                .clients
                .iter()
                .map(|c| EndpointId::from_bytes(c.id.get_p2p_public_key()).unwrap()),
        );
        Ok(new_state)
    }

    async fn send_witness(&mut self, opportunistic_data: OpportunisticData) -> Result<()> {
        Ok(self
            .tx
            .send(ToSend::Witness(Box::new(opportunistic_data)))?)
    }

    async fn send_health_check(&mut self, health_checks: HealthChecks<ClientId>) -> Result<()> {
        self.tx.send(ToSend::HealthCheck(health_checks))?;
        Ok(())
    }

    async fn send_checkpoint(&mut self, checkpoint: model::HubRepo) -> Result<()> {
        self.tx.send(ToSend::Checkpoint(checkpoint))?;
        Ok(())
    }

    async fn send_schema(&mut self, schema: ModelSchemaInfo) -> Result<()> {
        self.tx.send(ToSend::Schema(schema))?;
        Ok(())
    }
}

pub struct App {
    run_id: String,
    cancel: CancellationToken,
    update_tui_interval: Interval,
    tx_tui_state: Option<Sender<TabsData>>,
    coordinator_state: Coordinator<ClientId>,
    server_conn: TcpClient<ClientId, ClientToServerMessage, ServerToClientMessage>,
    capabilities: ClientCapabilities,
    training_assignment: Option<TrainingAssignment>,

    metrics: Arc<ClientMetrics>,
}

pub async fn build_app(
    cancel: CancellationToken,
    server_addr: String,
    tx_tui_state: Option<Sender<TabsData>>,
    p: TrainArgs,
) -> Result<(
    App,
    allowlist::AllowDynamic,
    NC,
    RunInitConfig<ClientId, ClientId>,
)> {
    let metrics = Arc::new(ClientMetrics::new(p.metrics_local_port));
    let identity_secret_key = read_identity_secret_key(p.identity_secret_key_path.as_ref())?
        .unwrap_or_else(|| SecretKey::generate(&mut rand::rng()));
    let server_conn = TcpClient::<ClientId, ClientToServerMessage, ServerToClientMessage>::connect(
        &server_addr,
        identity_secret_key.public().into(),
        identity_secret_key.clone(),
    )
    .await?;

    let hub_read_token = std::env::var("HF_TOKEN").ok();
    let eval_tasks = p.eval_tasks()?;
    let checkpoint_config = p.checkpoint_config()?;
    let wandb_info = p.wandb_info(format!(
        "{}-{}",
        p.run_id.clone(),
        identity_secret_key.public().fmt_short()
    ))?;

    let allowlist = allowlist::AllowDynamic::new();

    let p2p = NC::init(
        &p.run_id,
        p.bind_p2p_port,
        p.bind_p2p_interface,
        p.iroh_discovery,
        p.iroh_relay,
        vec![],
        Some(identity_secret_key.clone()),
        allowlist.clone(),
        metrics.clone(),
        Some(cancel.clone()),
    )
    .await?;

    let mut suffix_gate_start_step = p.matformer_suffix_gate_start_step;
    let mut suffix_gate_warmup_steps = p.matformer_suffix_gate_warmup_steps;
    let mut distill_start_step = p.matformer_distillation_start_step;
    let mut distill_warmup_steps = p.matformer_distillation_warmup_steps;
    if p.matformer_synergy_phase_a_steps > 0 || p.matformer_synergy_ramp_steps > 0 {
        if p.matformer_synergy_phase_a_steps == 0 || p.matformer_synergy_ramp_steps == 0 {
            return Err(Error::msg(
                "MatFormer synergy schedule requires both --matformer-synergy-phase-a-steps and --matformer-synergy-ramp-steps to be > 0",
            ));
        }
        suffix_gate_start_step = p.matformer_synergy_phase_a_steps;
        suffix_gate_warmup_steps = p.matformer_synergy_ramp_steps;
        distill_start_step = p.matformer_synergy_phase_a_steps;
        distill_warmup_steps = p.matformer_synergy_ramp_steps;
        info!(
            phase_a_steps = p.matformer_synergy_phase_a_steps,
            ramp_steps = p.matformer_synergy_ramp_steps,
            "Applying MatFormer synergy schedule overrides (suffix gate + distillation)"
        );
    }

    let state_options: RunInitConfig<ClientId, ClientId> = RunInitConfig {
        data_parallelism: p.data_parallelism,
        tensor_parallelism: p.tensor_parallelism,
        micro_batch_size: p.micro_batch_size,
        same_batch_warmup_steps: p.same_batch_warmup_steps,
        same_batch_calibration_every_steps: p.same_batch_calibration_every_steps,
        same_batch_calibration_start_step: p.same_batch_calibration_start_step,
        same_batch_calibration_no_apply: p.same_batch_calibration_no_apply,
        matformer_local_inner_steps: p.matformer_local_inner_steps,
        write_gradients_dir: p.write_gradients_dir,
        eval_tasks,
        eval_task_max_docs: p.eval_task_max_docs,
        prompt_task: p.prompt_task,
        heldout_eval_config: (p.heldout_eval_batches > 0).then_some(
            psyche_client::HeldoutEvalConfig {
                batches: p.heldout_eval_batches,
                batch_size: p.heldout_eval_batch_size,
            },
        ),
        checkpoint_config,
        hub_read_token,
        hub_max_concurrent_downloads: p.hub_max_concurrent_downloads,
        wandb_info,
        identity: identity_secret_key.public().into(),
        network_identity: identity_secret_key.public().into(),
        private_key: identity_secret_key,
        optim_stats_every_n_steps: p.optim_stats_steps,
        grad_accum_in_fp32: p.grad_accum_in_fp32,
        dummy_training_delay_secs: p.dummy_training_delay_secs,
        max_concurrent_parameter_requests: p.max_concurrent_parameter_requests,
        device: p.device.clone(),
        matformer_tier: p.matformer_tier,
        matformer_load_strategy: p.matformer_load_strategy,
        matformer_helper_fraction: p.matformer_helper_fraction,
        matformer_helper_rotation_interval: p.matformer_helper_rotation_interval,
        log_memory_usage: p.log_memory_usage,
        sidecar_port: p.sidecar_port,
        suffix_gate_config: if suffix_gate_warmup_steps > 0 || p.matformer_suffix_gate_learnable {
            Some(psyche_modeling::SuffixGateConfig {
                gate_tier: p.matformer_suffix_gate_tier,
                start_step: suffix_gate_start_step,
                warmup_steps: suffix_gate_warmup_steps,
                schedule: match p.matformer_suffix_gate_schedule {
                    psyche_client::MatformerGateSchedule::Linear => {
                        psyche_modeling::SuffixGateSchedule::Linear
                    }
                    psyche_client::MatformerGateSchedule::Cosine => {
                        psyche_modeling::SuffixGateSchedule::Cosine
                    }
                },
                learnable: p.matformer_suffix_gate_learnable,
                learnable_init_logit: p.matformer_suffix_gate_learnable_init_logit,
            })
        } else {
            None
        },
        distillation_config: if p.matformer_distillation_beta_max > 0.0 {
            Some(psyche_modeling::DistillationConfig {
                top_k: p.matformer_distillation_top_k,
                logits_to_keep: p.matformer_distillation_logits_to_keep,
                temperature: p.matformer_distillation_temperature,
                beta_max: p.matformer_distillation_beta_max,
                start_step: distill_start_step,
                warmup_steps: distill_warmup_steps,
                combine_mode: match p.matformer_distillation_combine_mode {
                    psyche_client::MatformerDistillationCombineMode::Mix => {
                        psyche_modeling::DistillationCombineMode::Mix
                    }
                    psyche_client::MatformerDistillationCombineMode::Add => {
                        psyche_modeling::DistillationCombineMode::Add
                    }
                },
                min_teacher_topk_mass: p
                    .matformer_distillation_min_teacher_topk_mass
                    .or(p.matformer_distill_confidence_threshold),
                kd_q_topk_mass_floor: p.matformer_distillation_kd_q_topk_mass_floor,
                ..Default::default()
            })
        } else {
            None
        },
        distro_apply_mode: match p.distro_apply_mode {
            psyche_client::DistroApplyMode::Sign => psyche_modeling::DistroApplyMode::Sign,
            psyche_client::DistroApplyMode::Raw => psyche_modeling::DistroApplyMode::Raw,
        },
        distro_aggregate_mode: match p.distro_aggregate_mode {
            psyche_client::DistroAggregateMode::Legacy => {
                psyche_modeling::DistroAggregateMode::Legacy
            }
            psyche_client::DistroAggregateMode::DilocoLite => {
                psyche_modeling::DistroAggregateMode::DilocoLite
            }
        },
        distro_value_mode: match p.distro_value_mode {
            psyche_client::DistroValueMode::Auto => psyche_modeling::DistroValueMode::Auto,
            psyche_client::DistroValueMode::Sign => psyche_modeling::DistroValueMode::Sign,
            psyche_client::DistroValueMode::Raw => psyche_modeling::DistroValueMode::Raw,
        },
        distro_apply_only_trainer_index: p.distro_apply_only_trainer_index,
        distro_raw_config: psyche_modeling::DistroRawConfig {
            enabled: p.distro_raw_v2_enabled,
            norm_mode: match p.distro_raw_norm_mode {
                psyche_client::DistroRawNormMode::Off => psyche_modeling::DistroRawNormMode::Off,
                psyche_client::DistroRawNormMode::MatchPreL2 => {
                    psyche_modeling::DistroRawNormMode::MatchPreL2
                }
                psyche_client::DistroRawNormMode::MatchSignEquivalent => {
                    psyche_modeling::DistroRawNormMode::MatchSignEquivalent
                }
                psyche_client::DistroRawNormMode::MatchSignEquivalentNnz => {
                    psyche_modeling::DistroRawNormMode::MatchSignEquivalentNnz
                }
            },
            scale_multiplier: p.distro_raw_scale_multiplier,
            scale_max: p.distro_raw_scale_max,
            abs_clip_mult: p.distro_raw_abs_clip_mult,
            sign_equiv_mult: p.distro_raw_sign_equiv_mult,
            missing_sidecar_policy: match p.distro_raw_missing_sidecar_policy {
                psyche_client::DistroRawMissingSidecarPolicy::WarnOff => {
                    psyche_modeling::DistroRawMissingSidecarPolicy::WarnOff
                }
                psyche_client::DistroRawMissingSidecarPolicy::Fail => {
                    psyche_modeling::DistroRawMissingSidecarPolicy::Fail
                }
            },
        },
        distro_diloco_lite_config: psyche_modeling::DistroDilocoLiteConfig {
            outer_momentum: p.distro_diloco_outer_momentum,
            outer_lr_multiplier: p.distro_diloco_outer_lr_multiplier,
            trust_region_target: p.distro_diloco_trust_region_target,
            trust_region_max_scale: p.distro_diloco_trust_region_max_scale,
            tier_weight_cap: p.distro_diloco_tier_weight_cap,
        },
        distro_sign_error_feedback_config: psyche_modeling::DistroSignErrorFeedbackConfig {
            enabled: p.distro_error_feedback,
            decay: p.distro_ef_decay,
        },
        distro_tier_prox_config: psyche_modeling::DistroTierProxConfig {
            mu: p.matformer_prox_mu,
            prefix_only: p.matformer_prox_prefix_only,
        },
    };
    let app = App {
        cancel,
        tx_tui_state,
        update_tui_interval: interval(Duration::from_millis(150)),
        coordinator_state: Coordinator::zeroed(),
        server_conn,
        run_id: p.run_id,
        capabilities: ClientCapabilities {
            device: p.device.to_string(),
            matformer_tier: p.matformer_tier,
        },
        training_assignment: None,
        metrics,
    };
    Ok((app, allowlist, p2p, state_options))
}

impl App {
    pub async fn run(
        &mut self,
        allowlist: allowlist::AllowDynamic,
        p2p: NC,
        state_options: RunInitConfig<ClientId, ClientId>,
    ) -> Result<()> {
        // sanity checks
        if let Some(checkpoint_config) = &state_options.checkpoint_config {
            if let Some(hub_upload) = &checkpoint_config.hub_upload {
                let api = hf_hub::api::tokio::ApiBuilder::new()
                    .with_token(Some(hub_upload.hub_token.clone()))
                    .build()?;
                let repo_api = api.repo(Repo::new(
                    hub_upload.hub_repo.clone(),
                    hf_hub::RepoType::Model,
                ));
                if !repo_api.is_writable().await {
                    anyhow::bail!(
                        "Checkpoint upload repo {} is not writable with the passed API key.",
                        hub_upload.hub_repo
                    )
                }
            }
        }

        self.server_conn
            .send(ClientToServerMessage::Join {
                run_id: self.run_id.clone(),
                capabilities: self.capabilities.clone(),
            })
            .await?;

        let (tx_from_server_message, rx_from_server_message) = mpsc::unbounded_channel();
        let (tx_to_server_message, mut rx_to_server_message) = mpsc::unbounded_channel();
        let mut client = Client::new(
            Backend {
                allowlist: allowlist.clone(),
                rx: rx_from_server_message,
                tx: tx_to_server_message,
            },
            allowlist,
            p2p,
            state_options,
            self.metrics.clone(),
        );

        debug!("Starting app loop");
        loop {
            select! {
                _ = self.cancel.cancelled() => {
                   break;
                }
                message = self.server_conn.receive() => {
                    self.on_server_message(message?, &tx_from_server_message).await;
                }
                _ = self.update_tui_interval.tick() => {
                    let (client_tui_state, network_tui_state) = client.tui_states().await;
                    self.update_tui(client_tui_state, network_tui_state).await?;
                }
                res = client.finished() => {
                    res??;
                }
                Some(to_send) = rx_to_server_message.recv() => {
                    match to_send {
                        ToSend::Witness(witness) => self.server_conn.send(ClientToServerMessage::Witness(witness)).await?,
                        ToSend::HealthCheck(health_checks) => self.server_conn.send(ClientToServerMessage::HealthCheck(health_checks)).await?,
                        ToSend::Checkpoint(checkpoint) => self.server_conn.send(ClientToServerMessage::Checkpoint(checkpoint)).await?,
                        ToSend::Schema(schema) => self.server_conn.send(ClientToServerMessage::Schema { schema }).await?,
                    };
                }
            }
        }
        Ok(())
    }

    async fn update_tui(
        &mut self,
        client_tui_state: ClientTUIState,
        network_tui_state: NetworkTUIState,
    ) -> Result<()> {
        if let Some(tx_tui_state) = &self.tx_tui_state {
            let states = (
                client_tui_state,
                (&self.coordinator_state).into(),
                network_tui_state,
                Default::default(),
            );
            tx_tui_state.send(states).await?;
        }
        Ok(())
    }

    async fn on_server_message(
        &mut self,
        message: ServerToClientMessage,
        tx: &mpsc::UnboundedSender<Coordinator<ClientId>>,
    ) {
        match message {
            ServerToClientMessage::Coordinator(state) => {
                self.coordinator_state = *state;
                let _ = tx.send(*state);
            }
            ServerToClientMessage::TrainingAssignment { assignment } => {
                info!(
                    matformer_tier = assignment.matformer_tier,
                    "Received training assignment"
                );
                self.training_assignment = Some(assignment);
            }
        }
    }
}
