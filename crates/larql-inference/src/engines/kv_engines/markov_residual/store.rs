//! RsStore — per-layer residual buffer for MarkovResidualEngine.

use ndarray::{Array2, s};
use crate::attention::SharedKV;

/// Per-layer pre-attention residuals for all stored positions.
pub struct RsStore {
    pub stored: Vec<Array2<f32>>,
    pub cold_residuals: Option<Vec<Array2<f32>>>,
    pub cold_kv: Option<Vec<SharedKV>>,
    pub cold_abs_start: usize,
    pub next_position: usize,
    pub max_window: Option<usize>,
}

impl RsStore {
    pub fn memory_bytes(&self) -> usize {
        let hot: usize = self.stored.iter().map(|s| s.len() * 4).sum();
        let cold_res: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum()).unwrap_or(0);
        let cold_kv: usize = self.cold_kv.as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum()).unwrap_or(0);
        hot + cold_res + cold_kv
    }

    pub fn cold_bytes(&self) -> usize {
        let cold_res: usize = self.cold_residuals.as_ref()
            .map(|c| c.iter().map(|s| s.len() * 4).sum()).unwrap_or(0);
        let cold_kv: usize = self.cold_kv.as_ref()
            .map(|kv| kv.iter().map(|(k, v)| (k.len() + v.len()) * 4).sum()).unwrap_or(0);
        cold_res + cold_kv
    }

    pub fn window_tokens(&self) -> usize {
        self.stored.first().map_or(0, |s| s.shape()[0])
    }

    pub(crate) fn clip_layer(&mut self, layer: usize, cold: &mut Vec<Array2<f32>>) {
        let window = match self.max_window { Some(w) => w, None => return };
        let s = &self.stored[layer];
        let rows = s.shape()[0];
        if rows <= window { cold.push(Array2::zeros((0, s.shape()[1]))); return; }
        let start = rows - window;
        cold.push(s.slice(s![..start, ..]).to_owned());
        self.stored[layer] = s.slice(s![start.., ..]).to_owned();
    }
}
