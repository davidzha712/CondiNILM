# CondiNILM Architecture Visualization - File Index

> Upload each file separately to Gemini to generate WebGL interactive visualizations.
> Each file is self-contained with full context needed for that section.

## Content Sections (9 files)

| File | Section | Key Content |
|------|---------|-------------|
| `01_system_overview.md` | System Overview | High-level architecture, design principles, config params |
| `02_data_preprocessing.md` | Data Preprocessing | Raw CSV -> sliding windows -> model input tensors |
| `03_feature_engineering.md` | Feature Engineering | Temporal encoding, electrical features, FFT frequency features |
| `04_network_forward_pass.md` | Complete Forward Pass | All 12 stages with exact tensor shapes at every step |
| `05_dilated_conv_embedding.md` | Dilated Conv Embedding | ResUnit stack, receptive field analysis |
| `06_transformer_encoder.md` | Transformer Encoder | 3-layer encoder with FiLM integration |
| `07_multihead_attention.md` | Multi-Head Attention | 8-head attention, diagonal masking, Q/K/V computation |
| `08_film_modulation.md` | FiLM Modulation | Encoder FiLM + Decoder FiLM, condition features, parameter generation |
| `09_gate_and_device_heads.md` | Gate & Device Heads | Smoothstep gate, hard gate, sparse CNN, training/loss/inference |

## Mermaid Diagram Files (7 files)

| File | Diagram | Description |
|------|---------|-------------|
| `mermaid_01_system_architecture.md` | Overall Architecture | End-to-end system flowchart with all modules |
| `mermaid_02_data_pipeline.md` | Data Pipeline | Raw data -> preprocessing -> windowing -> features |
| `mermaid_03_attention_detail.md` | Attention Detail | Q/K/V projection, diagonal masking, multi-head merge |
| `mermaid_04_film_flow.md` | FiLM Conditioning | Condition extraction -> MLP -> gamma/beta -> modulation |
| `mermaid_05_gate_mechanism.md` | Gate Mechanism | Classification/regression branches, smoothstep, hard threshold |
| `mermaid_06_training_pipeline.md` | Training Pipeline | Multi-crop, forward, 7-component loss, PCGrad |
| `mermaid_07_inference_pipeline.md` | Inference Pipeline | Seq2Subseq sliding window, post-processing |

## How to Use with Gemini

1. Upload files one at a time (or in small groups)
2. Ask Gemini to generate a WebGL interactive visualization for each section
3. Each file contains:
   - Detailed text descriptions with exact tensor shapes
   - Pseudocode for key computations
   - Suggested visualization interactions
   - Color coding and layout hints
4. Combine all generated WebGL components into a single page
