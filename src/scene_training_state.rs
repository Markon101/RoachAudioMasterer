//! Atomic native training snapshots: model, both Adam states and schedule.
//! Adam provenance: https://arxiv.org/abs/1412.6980; persistence is project code.
use crate::scene_model::{Adam, Model};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::Path,
};

#[derive(Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub schema: String,
    pub recipe: String,
    pub model: Model,
    pub encoder_adam: Adam,
    pub head_adam: Adam,
    pub learning_rate: f32,
    pub prior_backend: String,
}
impl TrainingState {
    pub fn new(model: Model, encoder_adam: Adam, head_adam: Adam, backend: &str) -> Self {
        Self {
            schema: "scene-v2-training-state-v1".into(),
            recipe: "scene-v2-fixed-block-schedule-v1".into(),
            model,
            encoder_adam,
            head_adam,
            learning_rate: 0.001,
            prior_backend: backend.into(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        let m = &self.model;
        ensure!(
            self.schema == "scene-v2-training-state-v1"
                && self.recipe == "scene-v2-fixed-block-schedule-v1"
                && m.validate()
                && ((m.stage == "flow" && m.seed == 73 && m.prior_fingerprint.is_some())
                    || (m.stage == "deterministic"
                        && m.seed == 71
                        && m.prior_fingerprint.is_none()))
                && m.optimizer_steps <= 12000
                && m.training_examples == m.optimizer_steps.div_ceil(4)
                && m.training_seed_start
                    .checked_add(m.training_examples as u64)
                    .is_some()
                && self
                    .encoder_adam
                    .validate(m.encoder.weights.len(), m.optimizer_steps)
                && self
                    .head_adam
                    .validate(m.head.weights.len(), m.optimizer_steps)
                && self.learning_rate == 0.001
                && matches!(self.prior_backend.as_str(), "cpu" | "opencl"),
            "invalid native training state, optimizer shape/counter, or recipe"
        );
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        ensure!(!path.exists(), "training snapshot already exists");
        let temporary = path.with_extension("json.tmp");
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, self)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        drop(writer);
        fs::rename(&temporary, path)?;
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self> {
        ensure!(
            fs::metadata(path)?.len() <= 16 * 1024 * 1024,
            "training state too large"
        );
        let s: Self = serde_json::from_reader(BufReader::new(fs::File::open(path)?))?;
        s.validate()?;
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn step(s: &mut TrainingState) {
        let index = s.model.optimizer_steps;
        let gradient = |n: usize| {
            (0..n)
                .map(|i| ((i % 17) as f32 - 8.0) * 0.0001 * (index + 1) as f32)
                .collect::<Vec<_>>()
        };
        let eg = gradient(s.model.encoder.weights.len());
        let hg = gradient(s.model.head.weights.len());
        s.encoder_adam
            .update(&mut s.model.encoder.weights, &eg, s.learning_rate);
        s.head_adam
            .update(&mut s.model.head.weights, &hg, s.learning_rate);
        s.model.optimizer_steps += 1;
        s.model.training_examples = s.model.optimizer_steps.div_ceil(4);
    }
    #[test]
    fn disk_resume_is_exact_and_rejects_corrupt_state() -> Result<()> {
        let model = Model::new(71);
        let mut whole = TrainingState::new(
            model.clone(),
            Adam::new(model.encoder.weights.len()),
            Adam::new(model.head.weights.len()),
            "cpu",
        );
        for _ in 0..7 {
            step(&mut whole);
        }
        let path = std::env::temp_dir().join(format!(
            "highband-adam-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        whole.save(&path)?;
        assert!(whole.save(&path).is_err(), "a snapshot must not overwrite");
        let mut resumed = TrainingState::load(&path)?;
        fs::remove_file(path)?;
        for _ in 7..12 {
            step(&mut whole);
            step(&mut resumed);
        }
        assert_eq!(
            serde_json::to_vec(&whole)?,
            serde_json::to_vec(&resumed)?,
            "weights, moments and counters must resume bit-for-bit"
        );
        let mut invalid = serde_json::to_value(&resumed)?;
        invalid["head_adam"]["v"][0] = serde_json::json!(-1.0);
        assert!(
            serde_json::from_value::<TrainingState>(invalid)?
                .validate()
                .is_err(),
            "negative second moment accepted"
        );
        resumed.model.optimizer_steps += 1;
        assert!(
            resumed.validate().is_err(),
            "optimizer counter mismatch accepted"
        );
        resumed.model.optimizer_steps -= 1;
        resumed.head_adam = Adam::new(1);
        assert!(
            resumed.validate().is_err(),
            "optimizer shape mismatch accepted"
        );
        Ok(())
    }
}
