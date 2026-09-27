#!/usr/bin/env python3
"""Local Streamlit demo for scoring the first two seconds of an iSeismometer CSV."""

from pathlib import Path
import io

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "centrifuge_imbalance.keras"
SAMPLE_RATE_HZ = 60.0
WINDOW_SECONDS = 2.0
N_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)
RISK_THRESHOLD = 0.40
AXES = ["X", "Y", "Z"]

st.set_page_config(page_title="Centrifuge vibration screen", page_icon="〰️", layout="centered")
st.title("Centrifuge vibration screen")
st.caption("Upload an iSeismometer CSV to score its first 2 seconds.")

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH.name}")
    st.info("Run the notebook's final training cell first; it saves the model here.")
    st.stop()

@st.cache_resource
def load_model(path: str, modified_ns: int, size_bytes: int):
    # Include file metadata in the cache key so notebook retraining/re-saving
    # cannot silently leave the app serving the previous model object.
    return tf.keras.models.load_model(path, compile=False)

try:
    model_stat = MODEL_PATH.stat()
    model = load_model(str(MODEL_PATH), model_stat.st_mtime_ns, model_stat.st_size)
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

uploaded = st.file_uploader("Choose a seismometer CSV", type=["csv"])
if uploaded is not None:
    try:
        frame = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        missing = set(["Timestamp", *AXES]) - set(frame.columns)
        if missing:
            raise ValueError("CSV must have columns Timestamp, X, Y, Z; missing: " + ", ".join(sorted(missing)))
        frame = frame[["Timestamp", *AXES]].apply(pd.to_numeric, errors="coerce").dropna()
        frame = frame.sort_values("Timestamp").drop_duplicates("Timestamp")
        if len(frame) < 2:
            raise ValueError("Need at least two valid timestamped samples.")

        timestamps = frame["Timestamp"].to_numpy(dtype=float)
        relative_time = timestamps - timestamps[0]
        if np.any(np.diff(relative_time) <= 0):
            raise ValueError("Timestamps must increase strictly.")
        observed_fs = 1.0 / float(np.median(np.diff(relative_time)))
        target_time = np.arange(N_SAMPLES, dtype=float) / SAMPLE_RATE_HZ
        if relative_time[-1] < target_time[-1]:
            raise ValueError(
                f"The file contains only {relative_time[-1]:.3f} s after its first sample; "
                "at least 1.983 s is needed to form a 120-sample, 60 Hz window."
            )
        source_values = frame[AXES].to_numpy(dtype=float)
        waveform = np.column_stack([
            np.interp(target_time, relative_time, source_values[:, axis_index])
            for axis_index in range(3)
        ]).astype(np.float32)
        model_input = waveform[np.newaxis, :, :]

        if tuple(model.input_shape[1:]) != (N_SAMPLES, 3):
            raise ValueError(
                f"Model expects input shape {model.input_shape}; app provides (1, {N_SAMPLES}, 3). "
                "Use a model trained with the same 60 Hz, 120-sample preprocessing."
            )
        logits = np.asarray(model(model_input, training=False)).reshape(-1)
        if logits.size != 1:
            raise ValueError("Expected a single linear output logit from the saved model.")
        # Notebook model uses a linear final layer and BinaryCrossentropy(from_logits=True).
        score = float(tf.math.sigmoid(logits[0]).numpy())

        st.success("First two seconds prepared for scoring.")
        st.metric("Imbalance model score", f"{100 * score:.1f}%")
        if score > RISK_THRESHOLD:
            st.warning("Warning: model score exceeds 40%.")
        st.progress(min(max(score, 0.0), 1.0))
        st.caption(
            f"Input: {len(frame)} valid rows; estimated source rate {observed_fs:.2f} Hz; "
            "resampled to 60 Hz × 2 s (120 samples)."
        )
        st.dataframe(pd.DataFrame(waveform, columns=AXES).head(10), use_container_width=True)
    except Exception as exc:
        st.error(str(exc))
