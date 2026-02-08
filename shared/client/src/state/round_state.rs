use crate::{fetch_data::BatchIdSet, Finished, TrainingResult};

use psyche_coordinator::{
    Commitment, CommitteeProof, CommitteeSelection, WitnessBloom, WitnessProof,
};
use psyche_core::{BatchId, NodeIdentity};
use psyche_modeling::DistroResult;
use psyche_network::{Hash, TransmittableTeacherLogits};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex},
};

use super::types::PayloadState;

#[derive(Debug)]
pub struct TeacherLogitsDownloadState<T: NodeIdentity> {
    pub from: T,
    pub batch_id: BatchId,
    /// Expected payload hash from the broadcast commitment (`Commitment.data_hash`).
    pub expected_commitment_hash: [u8; 32],
}

pub struct RoundState<T: NodeIdentity> {
    pub height: u32,
    pub step: u32,
    pub sent_witness: bool,
    pub sent_finished: bool,
    pub downloads: Arc<Mutex<HashMap<psyche_network::Hash, PayloadState<T>>>>,
    #[allow(clippy::type_complexity)]
    pub results: HashMap<BatchId, Vec<(T, (Commitment, TrainingResult))>>,
    pub clients_finished: HashMap<T, Finished>,
    pub data_assignments: BTreeMap<BatchId, T>,
    pub blooms: Arc<Mutex<Option<(WitnessBloom, WitnessBloom)>>>,
    pub broadcasts: Vec<[u8; 32]>,
    pub committee_info: Option<(CommitteeProof, WitnessProof, CommitteeSelection)>,
    pub batch_ids_not_yet_trained_on: Arc<Mutex<Option<BatchIdSet>>>,
    pub self_distro_results: Vec<Vec<DistroResult>>,
    /// Teacher logits received from tier-0 clients, keyed by batch ID.
    /// Students use these for in-place distillation loss in the same step.
    pub teacher_logits: Arc<Mutex<HashMap<BatchId, Arc<TransmittableTeacherLogits>>>>,
    /// Pending teacher-logits downloads keyed by blob hash (for commitment verification).
    pub teacher_logits_downloads: Arc<Mutex<HashMap<Hash, TeacherLogitsDownloadState<T>>>>,
}

impl<T: NodeIdentity> RoundState<T> {
    pub fn new() -> Self {
        Self {
            height: 0,
            step: 0,
            sent_witness: false,
            sent_finished: false,
            downloads: Arc::new(Mutex::new(HashMap::new())),
            results: HashMap::new(),
            broadcasts: Vec::new(),
            clients_finished: HashMap::new(),
            data_assignments: BTreeMap::new(),
            blooms: Arc::new(Mutex::new(None)),
            committee_info: None,
            batch_ids_not_yet_trained_on: Arc::new(Mutex::new(None)),
            self_distro_results: vec![],
            teacher_logits: Arc::new(Mutex::new(HashMap::new())),
            teacher_logits_downloads: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn distro_result_blob_downloaded(&self, hash: &psyche_network::Hash) -> bool {
        self.downloads.lock().unwrap().contains_key(hash)
    }
}

impl<T: NodeIdentity> Default for RoundState<T> {
    fn default() -> Self {
        RoundState::new()
    }
}
