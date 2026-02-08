use psyche_coordinator::{get_batch_ids_for_node, Coordinator};
use psyche_core::{BatchId, NodeIdentity};
use psyche_data_provider::{DataProvider, TokenizedDataProvider};
use psyche_modeling::{Batch, BatchData, BatchDataCPU};
use psyche_network::AuthenticatableIdentity;
use std::{
    collections::{BTreeMap, HashSet},
    marker::PhantomData,
    sync::Arc,
    time::Duration,
};
use tokio::{
    sync::{mpsc, Mutex},
    task::JoinHandle,
    time::sleep,
};
use tracing::{debug, error, info, trace, trace_span, warn, Instrument};

pub type BatchStep = u32;
pub type BatchIdSet = HashSet<BatchId>;

const MAX_RETRIES: u32 = 7;
const BASE_DELAY_MS: u64 = 2000;

pub struct DataFetcher<T: NodeIdentity, A: AuthenticatableIdentity> {
    data_provider: Arc<Mutex<DataProvider<A>>>,
    active_fetch_task: Option<(BatchStep, JoinHandle<()>)>,
    buffer_size: usize,
    same_batch_warmup_steps: u32,
    same_batch_calibration_every_steps: u32,
    same_batch_calibration_start_step: u32,
    same_batch_warmup_started_logged: bool,
    same_batch_warmup_ended_logged: bool,
    same_batch_calibration_logged: bool,
    _phantom: PhantomData<T>,
}

impl<T: NodeIdentity, A: AuthenticatableIdentity + 'static> DataFetcher<T, A> {
    pub fn new(
        data_provider: DataProvider<A>,
        buffer_size: usize,
        same_batch_warmup_steps: u32,
        same_batch_calibration_every_steps: u32,
        same_batch_calibration_start_step: u32,
    ) -> Self {
        Self {
            data_provider: Arc::new(Mutex::new(data_provider)),
            active_fetch_task: None,
            buffer_size,
            same_batch_warmup_steps,
            same_batch_calibration_every_steps,
            same_batch_calibration_start_step,
            same_batch_warmup_started_logged: false,
            same_batch_warmup_ended_logged: false,
            same_batch_calibration_logged: false,
            _phantom: Default::default(),
        }
    }

    /// Returns a handle to the underlying data provider, for ad-hoc batch fetches.
    ///
    /// This is used for tier-0 teacher-logit production (fetching batches not assigned to self).
    pub fn data_provider_handle(&self) -> Arc<Mutex<DataProvider<A>>> {
        self.data_provider.clone()
    }

    pub fn fetch_data(
        &mut self,
        state: &Coordinator<T>,
        data_assignments: &BTreeMap<BatchId, T>,
        identity: &T,
    ) -> TrainingDataForStep {
        let step = state.progress.step;
        let same_batch_warmup_active =
            self.same_batch_warmup_steps > 0 && step <= self.same_batch_warmup_steps;
        let calibration_start = self.same_batch_calibration_start_step.max(1);
        let same_batch_calibration_active = self.same_batch_calibration_every_steps > 0
            && step >= calibration_start
            && (step - calibration_start) % self.same_batch_calibration_every_steps == 0;
        let same_batch_active = same_batch_warmup_active || same_batch_calibration_active;

        if self.same_batch_warmup_steps > 0 && !self.same_batch_warmup_started_logged {
            info!(
                "[data] same-batch warmup enabled: steps 1..={} will fetch the same canonical batch for all trainers",
                self.same_batch_warmup_steps
            );
            self.same_batch_warmup_started_logged = true;
        }
        if self.same_batch_warmup_steps > 0
            && !self.same_batch_warmup_ended_logged
            && step == self.same_batch_warmup_steps.saturating_add(1)
        {
            info!(
                "[data] same-batch warmup ended at step {}; fetching assigned batches normally",
                step
            );
            self.same_batch_warmup_ended_logged = true;
        }
        if self.same_batch_calibration_every_steps > 0 && !self.same_batch_calibration_logged {
            info!(
                "[data] same-batch calibration enabled: every {} steps starting at step {}",
                self.same_batch_calibration_every_steps, calibration_start
            );
            self.same_batch_calibration_logged = true;
        }

        let mut assigned_batch_ids = get_batch_ids_for_node(data_assignments, identity);
        trace!(
            name:"fetching_data_assignments",
            assigned_batch_ids = assigned_batch_ids
                .iter()
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(","),
            "Fetching data assignments..."
        );

        let (tx_next_sample, next_sample) = mpsc::channel(self.buffer_size);

        let canonical_batch_id = if same_batch_active {
            data_assignments.keys().next().copied()
        } else {
            None
        };

        if same_batch_calibration_active {
            info!(
                step = step,
                canonical_batch_id = ?canonical_batch_id,
                "Applying same-batch calibration batch for this step"
            );
        }

        if let Some((last_step, task)) = self.active_fetch_task.take() {
            trace!("Killing previous fetch task from step {last_step}.");
            task.abort(); // we don't need it anymore :)
        }

        self.active_fetch_task = Some((
            step,
            tokio::spawn({
                trace!("New fetch task for step {step} has been spawned");
                let data_provider = self.data_provider.clone(); // only one of these tasks will acquire the lock at once. once one dies, the lock is released for sure.
                let canonical_batch_id = canonical_batch_id;

                async move {
                    loop {
                        let assigned_batch_id = match assigned_batch_ids.pop() {
                                Some(assigned) => assigned,
                                None => {
                                    // out of assigned data!
                                    return;
                                }
                        };
                        let fetch_batch_id = canonical_batch_id.unwrap_or(assigned_batch_id);

                        let mut retry_count = 0;
                        let batch = loop {
                            match data_provider.lock().await.get_samples(fetch_batch_id).await {
                                Ok(batch) => break batch,
                                Err(err) if retry_count < MAX_RETRIES => {
                                    retry_count += 1;
                                    let delay_ms = BASE_DELAY_MS * (retry_count as u64 - 1);
                                    warn!(
                                        "Data fetch error for assigned_batch_id={} fetch_batch_id={} (attempt {}/{}): \"{:#}\". Retrying in {}ms",
                                        assigned_batch_id, fetch_batch_id, retry_count, MAX_RETRIES, err, delay_ms
                                    );
                                    sleep(Duration::from_millis(delay_ms)).await;
                                    continue;
                                }
                                Err(err) => {
                                    error!(
                                        "Data fetch failed for assigned_batch_id={} fetch_batch_id={} after {} attempts: {err:#}",
                                        assigned_batch_id, fetch_batch_id, MAX_RETRIES
                                    );
                                    return;
                                }
                            }
                        };

                        if tx_next_sample
                            .send(Batch {
                                id: assigned_batch_id,
                                data: BatchData::CPU(batch.into_iter().map(|batch| {
                                    BatchDataCPU {
                                        input_ids: batch.input_ids,
                                        labels: batch.labels,
                                        position_ids: batch.position_ids,
                                        sequence_lengths: batch.sequence_lengths,
                                    }
                                }).collect()),
                            })
                            .await
                            .is_err()
                        {
                            debug!("Data loop finished");
                            return;
                        }
                    }
                }
                .instrument(trace_span!("fetch_data"))
            }),
        ));

        TrainingDataForStep { step, next_sample }
    }
}

pub struct TrainingDataForStep {
    pub step: u32,
    pub next_sample: mpsc::Receiver<Batch>,
}
