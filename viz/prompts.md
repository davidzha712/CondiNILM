# Gemini WebGL Prompts - CondiNILM Architecture Visualization

> Copy the prompt for each file and paste it into Gemini along with the corresponding MD file.
> Each prompt is self-contained and optimized for generating a single interactive WebGL HTML page.

---

## Prompt for `01_system_overview.md`

```
Based on the attached architecture specification, generate a single self-contained HTML file with an interactive WebGL/Three.js visualization of the CondiNILM system overview.

Requirements:
1. LAYOUT: Horizontal left-to-right data flow pipeline on a dark (#1a1a2e) background.
2. STAGES: Render each processing stage as a 3D rounded box with:
   - Color coding: Input=blue(#4fc3f7), Embedding=cyan(#26c6da), Transformer=orange(#ffa726), FiLM=pink(#f48fb1), Gate=purple(#ce93d8), Output=green(#66bb6a)
   - Label text on each box showing the stage name
   - Tensor shape annotation floating above each box (e.g., "(B, 7, 480)")
3. CONNECTIONS: Animated glowing particles flowing along curved spline paths between stages, showing data flow direction.
4. INTERACTION:
   - Orbit controls (rotate, zoom, pan the 3D scene)
   - Hover over any stage box: box glows, tooltip shows full description and tensor shape
   - Click a stage box: camera smoothly flies to that stage, shows expanded detail card with parameters
   - "Play" button: auto-animates camera panning through all stages left to right
5. DEVICE FAN-OUT: After the Transformer block, show 5 device output branches fanning out vertically (Kettle, Microwave, Fridge, Dishwasher, Washer), each with its own color and label.
6. FiLM SIDE PATH: Show condition feature extraction as a separate path below the main flow, with dashed animated lines feeding into Transformer layers and device heads.
7. CONFIG PANEL: Top-right floating panel showing key config values (d_model=96, n_heads=8, n_layers=3, etc.) with a semi-transparent dark background.
8. RESPONSIVE: Full viewport, handle window resize.
9. PERFORMANCE: Use instanced rendering for repeated elements. Target 60fps.
10. Include a title "CondiNILM / NILMFormer Architecture" with a subtle glow effect.

Output a single HTML file with all JS/CSS inlined. Use Three.js from CDN.
```

---

## Prompt for `02_data_preprocessing.md`

```
Based on the attached data preprocessing specification, generate a single self-contained HTML file with an interactive WebGL visualization of the CondiNILM data preprocessing pipeline.

Requirements:
1. PIPELINE VIEW: Vertical top-to-bottom flow showing 8 preprocessing steps.
2. SIGNAL VISUALIZATION: At the top, render a realistic household power waveform (procedurally generated) as a 2D line chart on a WebGL canvas:
   - X-axis: time (480 steps = 8 hours), Y-axis: power (0-6000W)
   - Include random kettle spikes (~3000W, 3-5 min), fridge cycling (~120W, periodic), baseline noise
3. ANIMATED TRANSFORMATIONS: For each preprocessing step, animate the signal transformation:
   - Step 2 (Cutoff): Peaks above 6000W get clipped with a red flash
   - Step 3 (Normalize): Y-axis smoothly rescales from [0,6000] to [0,1]
   - Step 4 (State Detection): Binary ON/OFF overlay appears as colored bands (green=ON, gray=OFF)
   - Step 5 (Sliding Window): A glowing rectangle sweeps across the signal, leaving extracted windows stacking below
   - Step 6 (Temporal Encoding): Sin/cos waves appear as colored overlays (orange/purple)
4. TENSOR SHAPE DISPLAY: At each step, show the tensor shape as a 3D block:
   - Width=batch, Height=channels, Depth=time
   - Blocks morph between steps to show shape changes
5. INTERACTION:
   - Play/Pause button to control animation speed
   - Step-by-step mode: click arrows to advance one step at a time
   - Hover on any step card to see detailed description
   - Slider to control the sliding window position manually
6. WINDOW OVERLAP VISUALIZATION: Show multiple overlapping windows with transparency, color-coded for train(blue)/val(yellow)/test(green) splits.
7. Bottom: final tensor (B,7,480) shown as an interactive 3D colored block that can be rotated.

Output a single HTML file with all JS/CSS inlined. Use Three.js + a 2D chart library (draw directly on canvas).
```

---

## Prompt for `03_feature_engineering.md`

```
Based on the attached feature engineering specification, generate a single self-contained HTML file with an interactive WebGL visualization showing how CondiNILM extracts condition features.

Requirements:
1. THREE-PANEL LAYOUT:
   - Left panel: Input power waveform (480 timesteps, interactive)
   - Center panel: Feature extraction animation
   - Right panel: Resulting 13-dim condition vector as colored bar chart

2. ELECTRICAL FEATURES (top section):
   - Show the waveform with interactive overlays:
     - MEAN: horizontal dashed line at average value
     - STD: shaded band around mean (±1 std)
     - RMS: second horizontal line (slightly above mean)
     - PEAK: vertical arrow pointing to maximum, with label
     - CREST: ratio annotation between peak and RMS arrows
   - Animate each feature appearing one by one with a 0.5s delay

3. FFT FREQUENCY FEATURES (bottom section):
   - Show the time-domain waveform on the left
   - Animate FFT transformation: waveform "decomposes" into frequency components
   - Show magnitude spectrum as bar chart with 8 color-coded frequency bands
   - Band colors: gradient from warm (band 0, low freq) to cool (band 7, high freq)
   - Hover over a frequency band: highlight the corresponding time-domain pattern

4. SCENARIO SWITCHER: 3 buttons to toggle between scenarios:
   - "Kettle ON" (high peak, high crest, energy in band 0-1)
   - "Fridge Cycling" (low peak, periodic, energy in band 2-3)
   - "Idle" (very low everything, flat spectrum)
   Each scenario changes the waveform and all derived features animate to new values.

5. CONDITION VECTOR: 13-bar horizontal chart:
   - Bars 0-4: labeled electrical features (warm colors)
   - Bars 5-12: labeled frequency bands (cool colors)
   - Values update in real-time when scenario changes
   - Hover shows exact value and feature name

6. FLOW ARROW: Animated arrow from condition vector pointing to "→ FiLM Network" label, showing where these features go next.

Output a single HTML file with all JS/CSS inlined. Use WebGL for waveform rendering, canvas 2D for charts.
```

---

## Prompt for `04_network_forward_pass.md`

```
Based on the attached complete forward pass specification, generate a single self-contained HTML file with an interactive WebGL/Three.js visualization showing data flowing through all 12 stages of the NILMFormer network.

Requirements:
1. 3D TENSOR FLOW: Render each stage's tensor as a 3D rectangular cuboid:
   - X-axis = batch dimension (thin, since less important)
   - Y-axis = channel/feature dimension (height varies: 7→8→96→480→96→5)
   - Z-axis = sequence length (depth=480, constant for most stages)
   - Color: each stage has a distinct color from the specification's color scheme

2. MORPHING ANIMATION: Tensors smoothly morph between stages:
   - Stage 2 (DilatedConv): block height changes 7→8
   - Stage 4 (Projection): block height expands 8→96
   - Stage 5 (Transpose): block rotates 90° (height↔depth swap)
   - Stage 7 (Encoder): block pulses 3 times (one per layer)
   - Stage 9 (Transpose back): block rotates 90° back
   - Stage 10-11 (Device heads): block splits into 5 thinner blocks

3. CAMERA PATH: Smooth camera dolly following the data flow:
   - Auto-play mode: camera tracks alongside the flowing data
   - Manual mode: scroll to scrub through stages

4. ANNOTATIONS: For each stage, floating text labels showing:
   - Stage name (e.g., "STAGE 4: Input Projection")
   - Tensor shape (e.g., "(32, 96, 480)")
   - Operation description (e.g., "Conv1d(8, 96, k=1)")

5. STAGE SELECTOR: Left sidebar with 12 clickable stage buttons. Clicking a stage:
   - Camera flies to that stage
   - Detail panel appears showing the full computation description
   - Tensor dimensions highlighted with ruler lines

6. PARTICLE SYSTEM: Colored particles flow through the tensors showing data movement. Particle color matches the current stage's color.

7. PARAMETER COUNTER: Running counter at bottom showing cumulative parameters at each stage.

Output a single HTML file with all JS/CSS inlined. Use Three.js from CDN. Target 60fps.
```

---

## Prompt for `05_dilated_conv_embedding.md`

```
Based on the attached dilated convolution specification, generate a single self-contained HTML file with an interactive WebGL visualization of the DilatedBlock embedding layer.

Requirements:
1. SIGNAL INPUT: Top of the screen shows a procedural 480-timestep power waveform with 7 channels rendered as colored lines.

2. DILATED KERNEL VISUALIZATION: For each of the 4 ResUnit layers:
   - Show the 3-tap convolution kernel positioned on the signal
   - Animate the kernel sliding across the time axis
   - Draw DILATION GAPS as dotted vertical lines connecting kernel taps to their input positions:
     - Layer 0 (d=1): taps at [t-1, t, t+1] (no gaps)
     - Layer 1 (d=2): taps at [t-2, t, t+2] (1 gap between each)
     - Layer 2 (d=4): taps at [t-4, t, t+4] (3 gaps)
     - Layer 3 (d=8): taps at [t-8, t, t+8] (7 gaps)
   - Kernel taps glow when active

3. RECEPTIVE FIELD GROWTH ANIMATION:
   - For a selected position t=240, show the receptive field expanding:
     - After layer 0: 3 positions highlighted in blue
     - After layer 1: 7 positions highlighted in cyan
     - After layer 2: 15 positions highlighted in green
     - After layer 3: 31 positions highlighted in yellow
   - Animate the expansion as a pulsing glow that grows outward
   - Show numeric label: "Receptive Field: 31 timesteps = 31 minutes"

4. FEATURE MAP GALLERY: Below the kernel visualization:
   - 8 horizontal heatmap strips (one per output channel)
   - Aligned with the input signal for comparison
   - Hover over a feature map position: highlight the receptive field on the input

5. RESIDUAL CONNECTION: Show the skip connection as a parallel path:
   - Main path: Conv → GELU → BatchNorm
   - Skip path: straight arrow (identity if channels match, 1x1 conv if not)
   - Addition point: two streams merge with a "+" symbol

6. LAYER SELECTOR: 4 tabs (Layer 0-3) to focus on one layer at a time. Active layer is fully rendered, others are dimmed.

7. GELU ACTIVATION PLOT: Small inset showing the GELU curve with current input/output value highlighted.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `06_transformer_encoder.md`

```
Based on the attached Transformer encoder specification, generate a single self-contained HTML file with an interactive WebGL/Three.js visualization of the 3-layer Transformer encoder with FiLM integration.

Requirements:
1. STACKED LAYERS: Render 3 encoder layers as vertically stacked translucent 3D blocks. Each block contains:
   - LayerNorm (thin blue band)
   - Attention (orange block with internal head visualization)
   - Residual connection (curved arrow bypassing the block)
   - LayerNorm (thin blue band)
   - FFN (green block that expands then contracts: 96→384→96)
   - FiLM modulation (pink highlight on FFN output)
   - Residual connection (curved arrow)

2. EXPANDABLE LAYERS: Click a layer to expand it vertically, revealing internal sub-components with animated data flow. Other layers compress to make room.

3. FFN EXPANSION ANIMATION:
   - Show the 96-dim feature vector as a row of colored bars
   - Animate expansion: bars grow to 384 (4x wider display)
   - GELU activation: bars get nonlinearly transformed (some flip, some shrink)
   - Compression: bars shrink back to 96
   - FiLM: bars get scaled (grow/shrink) and shifted (move up/down) by gamma/beta

4. FiLM OVERLAY: For each layer, show a side panel with:
   - gamma values as a horizontal bar chart (96 bars, range [-0.5, 0.5])
   - beta values as another bar chart
   - Color: positive=red, negative=blue, zero=gray
   - Label: "FiLM Layer {n}: (1 + γ)·x + β"

5. FEATURE EVOLUTION: Show how a single position's 96-dim vector changes through 3 layers:
   - 3 stacked bar charts (one per layer output)
   - Color intensity shows magnitude
   - Animated transition between layers

6. LAYER COMPARISON: Toggle to see all 3 layers side-by-side instead of stacked.

7. ATTENTION HEATMAP INSET: Small 480×480 heatmap thumbnail for each layer (click to expand to full Section 7 detail).

Output a single HTML file with all JS/CSS inlined. Use Three.js from CDN.
```

---

## Prompt for `07_multihead_attention.md`

```
Based on the attached multi-head attention specification, generate a single self-contained HTML file with an interactive WebGL visualization of the 8-head diagonally-masked self-attention mechanism.

Requirements:
1. TOP SECTION - Q/K/V PROJECTION:
   - Show input vector (96 colored bars) on the left
   - Three weight matrices (Wq, Wk, Wv) as small heatmap blocks
   - Three output vectors (Q, K, V) on the right, each a different color
   - Animate: input flows through each matrix to produce Q, K, V

2. MULTI-HEAD SPLIT:
   - Animate the 96-bar Q vector splitting into 8 groups of 12 bars
   - Each group slides into its own "lane" (color-coded by head: head0=red, head1=orange, head2=yellow, head3=green, head4=cyan, head5=blue, head6=indigo, head7=violet)
   - Same for K and V

3. ATTENTION MATRIX (MAIN FEATURE):
   - Large interactive heatmap showing one head's 480×480 (or simplified 32×32) attention matrix
   - DIAGONAL MASKING TOGGLE: Button to turn diagonal masking on/off
     - OFF: diagonal is brightest (self-attention dominates)
     - ON: diagonal turns BLACK (zero), other values intensify
   - Head selector: 8 buttons to switch between heads
   - Row highlight: Click a query position → highlight that row, show attention distribution as bar chart below
   - Color scale: dark blue (0) → white (0.5) → bright red (1.0)

4. DIAGONAL MASKING ANIMATION:
   - Step 1: Show raw attention scores (diagonal highest)
   - Step 2: Animate diagonal cells turning red with -10000 values
   - Step 3: Softmax: matrix values reorganize, diagonal becomes 0
   - Step 4: Zero cleanup: diagonal cells turn completely black

5. VALUE AGGREGATION:
   - Select a query position (e.g., t=42)
   - Show attention weights as a histogram over all 480 positions
   - Below: the top-5 attended positions' V vectors
   - Animate the weighted sum producing the output vector

6. HEAD MERGE: 8 colored 12-dim bars converge into one 96-dim bar, then pass through Wo projection.

7. INSIGHT PANEL: Text overlay explaining: "Diagonal masking forces the model to use CONTEXT rather than copying input. Critical for disaggregation where input=aggregate, output=per-device."

Output a single HTML file with all JS/CSS inlined. Use WebGL for heatmap rendering, canvas 2D for charts.
```

---

## Prompt for `08_film_modulation.md`

```
Based on the attached FiLM modulation specification, generate a single self-contained HTML file with an interactive WebGL visualization of the Feature-wise Linear Modulation mechanism.

Requirements:
1. THREE-COLUMN LAYOUT:
   - Left: Condition Feature Extraction
   - Center: FiLM Parameter Generation (Encoder + Decoder)
   - Right: Modulation Application

2. LEFT COLUMN - CONDITION FEATURES:
   - Input waveform at top (interactive, drag to change)
   - 5 electrical feature gauges (circular meters showing mean, std, rms, peak, crest)
   - FFT magnitude spectrum as 8-band bar chart
   - Combined 13-dim condition vector as horizontal bar chart
   - All update in real-time when waveform changes

3. CENTER COLUMN - PARAMETER GENERATION:
   Split into two parallel paths:

   ENCODER FiLM (top):
   - Show condition (13 bars) + device embedding (32 bars) concatenating into 45 bars
   - MLP animation: 45 → 32 (ReLU) → 576
   - Reshape animation: flat 576 → 3D grid (5 devices × 3 layers × 2×96)
   - tanh activation: values constrained to [-0.5, 0.5] with curve visualization
   - Average over devices: 5 rows → 1 row

   DECODER FiLM (bottom):
   - Same concat animation
   - MLP: 45 → 32 → 2
   - Split into gamma and beta scalars per device
   - 5 devices shown as 5 pairs of gamma/beta bars

4. RIGHT COLUMN - MODULATION APPLICATION:
   - Show a feature vector (96 bars) labeled "FFN Output"
   - Gamma as overlay arrows (↑ if positive = amplify, ↓ if negative = dampen)
   - Beta as offset arrows (shift bars up/down)
   - Animated transformation: bars grow/shrink by (1+gamma) then shift by beta
   - Before/After comparison toggle

5. DEVICE SELECTOR: 5 device tabs (Kettle, Microwave, Fridge, Dishwasher, Washer)
   - Switching device changes the embedding, which changes gamma/beta
   - Animate the cascade: new embedding → new MLP output → new modulation

6. FORMULA OVERLAY: Large floating formula "(1 + γ) × x + β" with animated arrows showing what gamma and beta do.

7. INTERACTIVE SLIDERS: Manual gamma slider [-0.5, 0.5] and beta slider [-0.5, 0.5] to see real-time modulation effect on a sample feature vector.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `09_gate_and_device_heads.md`

```
Based on the attached gate mechanism and device heads specification, generate a single self-contained HTML file with an interactive WebGL visualization covering the gate mechanism, device heads, loss functions, and inference pipeline.

Requirements:
1. FOUR-TAB INTERFACE: Tab bar at top with 4 sections:

TAB 1 - DEVICE HEAD ARCHITECTURE:
   - Show SimpleDeviceHead as a flow diagram:
     - Shared Conv block (96→128→128) as blue box
     - Two branches splitting from it:
       - Classification (red path): Conv→2×sigmoid→gate probability
       - Regression (blue path): Conv→ReLU→power magnitude
     - Merge point: gate × power
   - Show SparseDeviceCNN as simplified CNN stack
   - 5 device tabs to switch between head types

TAB 2 - GATE MECHANISM:
   - Interactive function plot showing THREE curves on same axes:
     - Sigmoid (gray, dashed): for comparison
     - Smoothstep (green, solid): s²(3-2s), the training gate
     - Hard threshold (orange, step): the inference gate
   - Interactive threshold slider (0.3-0.7): moves the step function
   - Below: Time-series example showing fridge cycling:
     - Top row: gate probability over 480 timesteps
     - Middle row: raw power prediction
     - Bottom row: gated output (toggle soft/hard mode)
   - Animated particles showing signal flowing through the gate

TAB 3 - LOSS FUNCTION:
   - Stacked bar chart showing 7 loss components for each of 5 devices
   - Interactive weight sliders (α_on, α_off, w_peak, w_grad, w_energy, λ_zero, λ_off)
   - Real-time loss recalculation when weights change
   - PCGrad visualization: 2D vector space with 5 gradient arrows
     - Animate conflict detection and projection
     - Toggle PCGrad on/off to see difference

TAB 4 - INFERENCE PIPELINE:
   - Top: Full 24-hour signal (scrollable)
   - Animated sliding window sweeping across:
     - Window boundary shown as rectangle
     - Center region highlighted in green
     - Margins in gray
   - Stitched output building below
   - Post-processing layers:
     - Toggle "short activation suppression" on/off
     - Toggle "long OFF gate suppression" on/off
     - Show removed segments highlighted in red
   - Final output: 5 stacked device signals in actual Watts

2. GLOBAL: Dark theme (#1a1a2e background), smooth transitions between tabs, responsive layout.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_01_system_architecture.md`

```
Based on the attached Mermaid diagram of the CondiNILM system architecture, generate a single self-contained HTML file with an interactive WebGL visualization that brings this diagram to life.

Requirements:
1. Convert the Mermaid flowchart into an interactive 3D node-graph:
   - Each node is a 3D rounded rectangle with the label text
   - Edges are animated 3D tubes with glowing particles flowing along them
   - Subgraphs rendered as transparent bounding boxes with labels

2. COLOR CODING from the diagram:
   - Input: blue (#e1f5fe)
   - Output: green (#c8e6c9)
   - Condition: orange (#fff3e0)
   - FiLM: pink (#fce4ec)
   - Attention: indigo (#e8eaf6)

3. INTERACTION:
   - Orbit controls to rotate/zoom the 3D graph
   - Click any node: fly-to animation, show description tooltip
   - Hover: node glows, edge particles speed up
   - Double-click a subgraph: zoom into that section

4. TENSOR SHAPE ANNOTATIONS: Each edge has a floating label showing the tensor shape (from the specification).

5. ANIMATED DATA FLOW: Particles continuously flow from Input through the graph to Output, splitting at branches and merging at concatenation.

6. MINIMAP: Small 2D overview in bottom-right corner showing the full graph with current viewport highlighted.

Output a single HTML file with all JS/CSS inlined. Use Three.js from CDN.
```

---

## Prompt for `mermaid_02_data_pipeline.md`

```
Based on the attached Mermaid diagram of the data preprocessing pipeline, generate a single self-contained HTML file with an interactive visualization.

Requirements:
1. VERTICAL PIPELINE: 8 processing steps as cards flowing top to bottom.
2. Each card has:
   - Step number and name
   - Mini visualization of the transformation (inline canvas)
   - Tensor shape badge
   - Expand button to show details
3. ANIMATED SIGNAL: A procedural power waveform flows through the pipeline:
   - At each step, the signal visually transforms (clip, normalize, window, encode)
   - Smooth CSS transitions between states
4. TENSOR SHAPE EVOLUTION: Right sidebar showing shapes as 3D blocks morphing at each step.
5. STEP-BY-STEP MODE: Buttons to advance one step at a time, with the current transformation highlighted.
6. INTERACTIVE WINDOW SLIDER: In Step 6, drag a slider to move the sliding window across the signal.
7. COLOR-CODED SPLITS: Train=blue, Val=yellow, Test=green regions.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_03_attention_detail.md`

```
Based on the attached Mermaid diagram of multi-head attention, generate a single self-contained HTML file with an interactive attention visualization.

Requirements:
1. FLOW: Left-to-right data flow matching the Mermaid diagram layout.
2. Q/K/V PROJECTION: Animated matrix multiplication visualization.
3. MULTI-HEAD SPLIT: 96-dim bar splits into 8 color-coded groups.
4. ATTENTION HEATMAP: Interactive 32×32 (simplified from 480×480) attention matrix:
   - Toggle diagonal masking on/off with animated transition
   - Click a row to see its attention distribution
   - Head selector (8 buttons)
5. VALUE AGGREGATION: Top-K visualization showing which positions contribute most.
6. HEAD MERGE: 8 streams converging into one output.
7. SMOOTH ANIMATIONS between all steps with data particles.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_04_film_flow.md`

```
Based on the attached Mermaid diagram of FiLM conditioning, generate a single self-contained HTML file with an interactive FiLM visualization.

Requirements:
1. DUAL-PATH LAYOUT: Encoder FiLM (top) and Decoder FiLM (bottom) as parallel flows.
2. CONDITION EXTRACTION: Animated signal analysis producing 13-dim vector.
3. DEVICE EMBEDDING: 5 color-coded embedding vectors (one per appliance).
4. MLP ANIMATION: Show concatenation → fc1 → ReLU → fc2 → reshape as animated bar transformations.
5. TANH BOUNDING: Interactive plot of 0.5×tanh(x) showing how values get bounded.
6. MODULATION PREVIEW: For a sample feature vector, show before/after FiLM modulation.
7. DEVICE SELECTOR: Switch between devices to see different gamma/beta values.
8. NUMERICAL EXAMPLE: Floating panel showing the concrete example from the specification.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_05_gate_mechanism.md`

```
Based on the attached Mermaid diagram of the gate mechanism, generate a single self-contained HTML file with an interactive gate visualization.

Requirements:
1. SPLIT VIEW:
   - Left: SimpleDeviceHead architecture diagram (shared features → two branches → merge)
   - Right: Interactive gate function plots

2. FUNCTION COMPARISON: Three curves on same axes:
   - Smoothstep: s²(3-2s) in green
   - Hard threshold: step function in orange
   - Sigmoid: for reference in gray
   - Interactive x-axis cursor showing current input value and all three outputs

3. TIMELINE EXAMPLE: Fridge cycling data (from specification):
   - Top: gate probability timeline (color gradient red→green)
   - Middle: raw power prediction (blue line)
   - Bottom: gated output with toggle between soft(training)/hard(inference)

4. SPARSE DEVICE CNN: Separate small diagram for kettle/microwave CNN bypass.

5. THRESHOLD SLIDER: Drag to change hard gate threshold (0.3-0.7), see output change in real-time.

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_06_training_pipeline.md`

```
Based on the attached Mermaid diagram of the training pipeline, generate a single self-contained HTML file with an interactive training visualization.

Requirements:
1. TRAINING LOOP FLOW: Animated flowchart showing data → model → loss → gradients → update cycle.

2. LOSS COMPONENT DASHBOARD:
   - 5 columns (one per device)
   - 7 rows (one per loss component)
   - Each cell is a colored bar showing the loss value
   - Interactive: adjust weights with sliders, see total loss change

3. PCGRAD ANIMATION:
   - 2D canvas showing gradient vectors as arrows
   - Animate conflict detection: when cos(g_i, g_j) < 0, arrows flash red
   - Animate projection: conflicting arrow gets projected, new direction shown in green
   - Step-by-step with play/pause

4. DEVICE TYPE CARDS: 4 cards (sparse_high_power, cycling_low_power, long_cycle, always_on) each showing:
   - Device names
   - Loss parameter values as a radar/spider chart
   - Distinctive color

5. TRAINING SCHEDULE TIMELINE: Bottom bar showing:
   - Warmup phase → main training phase
   - Anti-collapse scale curve (1.0→0.2)
   - Output stats alpha curve (0.0→1.0)
   - Animated playhead scrubbing through epochs

Output a single HTML file with all JS/CSS inlined.
```

---

## Prompt for `mermaid_07_inference_pipeline.md`

```
Based on the attached Mermaid diagram of the inference pipeline, generate a single self-contained HTML file with an interactive inference visualization.

Requirements:
1. MAIN ANIMATION - SLIDING WINDOW:
   - Top: Full 24-hour power signal (procedurally generated, scrollable)
   - Animated sliding window (480-wide rectangle) sweeping left to right
   - Window interior: center region highlighted green, margins gray
   - Below the signal: stitched output building up as centers accumulate
   - Play/Pause button, speed control

2. WINDOW DETAIL VIEW: When window is selected:
   - Show the full 480 timesteps of that window
   - Margin regions (120 each) dimmed
   - Center region (240) bright with border
   - Model output overlaid

3. POST-PROCESSING STAGES (toggleable):
   - Stage 1: Short activation suppression
     - Show removed short segments highlighted in red, then fading out
   - Stage 2: Long OFF gate suppression
     - Show gate probability as heat overlay
     - Suppressed regions highlighted in blue, then zeroed
   - Stage 3: Denormalization
     - Y-axis labels change from normalized to Watts

4. FINAL DISAGGREGATION DISPLAY:
   - Top: aggregate input signal (gray)
   - Below: 5 stacked device output signals, each in its own color:
     - Kettle (red), Microwave (orange), Fridge (cyan), Dishwasher (blue), Washer (purple)
   - Hover: tooltip showing exact Watts at any timepoint
   - Conservation check: sum line overlay (should approximate aggregate)

5. COMPUTATION STATS: Bottom panel showing:
   - Number of windows processed
   - Total FLOPs
   - Estimated time on GPU vs CPU

Output a single HTML file with all JS/CSS inlined.
```

---

## General Tips for All Prompts

When pasting these into Gemini, you can add these universal instructions:

```
Additional instructions:
- Make the visualization responsive and full-viewport
- Use a dark theme with (#1a1a2e) background and light text (#e0e0e0)
- Add smooth CSS/WebGL transitions (300ms ease-in-out)
- Include a "Help" button showing keyboard shortcuts and interaction guide
- Add loading spinner while WebGL initializes
- Ensure the HTML file works offline (only CDN dependencies: Three.js, if needed)
- Target 60fps performance
- Use anti-aliasing for crisp rendering
- Include a small "CondiNILM" watermark in the bottom-left corner
```
