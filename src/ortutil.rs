//! Small helpers around the `ort` API (session construction, tensor I/O).

use anyhow::{anyhow, Result};
use ndarray::ArrayD;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;
use ort::value::DynValue;
use std::path::Path;

/// Build an ONNX session with sane CPU settings.
pub fn make_session(path: &Path, intra_threads: usize) -> Result<Session> {
    let mut builder: SessionBuilder = Session::builder().map_err(|e| anyhow!("ort builder: {e}"))?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?;
    builder = builder.with_intra_threads(intra_threads).map_err(|e| anyhow!("{e}"))?;
    builder
        .commit_from_file(path)
        .map_err(|e| anyhow!("loading {}: {e}", path.display()))
}

/// Extract an f32 tensor as an owned dynamic ndarray.
pub fn value_to_f32_array(v: &DynValue) -> Result<ArrayD<f32>> {
    let (shape, data) = v.try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(ndarray::Array::from_shape_vec(dims, data.to_vec())?)
}
