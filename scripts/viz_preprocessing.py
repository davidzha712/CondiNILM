import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.helpers.dataset import NILMscaler, NILMDataset
from src.helpers.preprocessing import UKDALE_DataBuilder, REFIT_DataBuilder
from src.baselines.nilm.unetnilm import UNetNiLM

def load_real_data(dataset_name, house_idx, appliance, window_size, sampling_rate="1min"):
    """
    Load real data using the project's DataBuilders.
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    
    try:
        if dataset_name == "UKDALE":
            data_path = os.path.join(base_path, "UKDALE")
            # UKDALE_DataBuilder expects the directory containing house_X folders
            
            db = UKDALE_DataBuilder(
                data_path=data_path,
                mask_app=appliance,
                sampling_rate=sampling_rate,
                window_size=window_size
            )
            
            with st.spinner(f"Loading UKDALE House {house_idx} data..."):
                X, st_date = db.get_nilm_dataset(house_indicies=[house_idx])
                return X, st_date

        elif dataset_name == "REFIT":
            data_path = os.path.join(base_path, "REFIT", "RAW_DATA_CLEAN")
            
            db = REFIT_DataBuilder(
                data_path=data_path,
                mask_app=appliance,
                sampling_rate=sampling_rate,
                window_size=window_size
            )
            
            with st.spinner(f"Loading REFIT House {house_idx} data..."):
                X, st_date = db.get_nilm_dataset(house_indicies=[house_idx])
                return X, st_date
                
    except FileNotFoundError as e:
        st.error(f"File Not Found: {e}")
        st.warning(f"Please ensure {dataset_name} data is placed in {base_path}")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    
    return None, None

def generate_synthetic_data(length=10000, window_size=256):
    t = np.arange(length)
    # Appliance: Period 500, Duty cycle 0.3
    appliance_power = np.zeros(length)
    period = 500
    on_duration = 150
    
    for i in range(0, length, period):
        end = min(i + on_duration, length)
        appliance_power[i:end] = 1000 + np.random.normal(0, 50, end - i) # 1000W + noise

    # Background load
    background = 200 + np.random.normal(0, 20, length)
    
    # Aggregate (Mains)
    aggregate_power = appliance_power + background
    
    return t, aggregate_power, appliance_power

def run_preprocessing_tab():
    st.header("NILM Preprocessing Pipeline Visualization")
    st.markdown("Visualizing data transformation from Raw Waveform to Model Input.")

    # Sidebar Controls for Preprocessing
    st.sidebar.header("Preprocessing Configuration")
    
    data_source = st.sidebar.radio("Data Source (Preprocessing)", ["Synthetic", "Real Data (Local)"], key="prep_source")
    
    if data_source == "Synthetic":
        length = st.sidebar.slider("Data Length", 1000, 20000, 5000, key="prep_len")
        window_size = st.sidebar.slider("Window Size", 64, 512, 256, key="prep_win")
        sampling_rate = st.sidebar.selectbox("Sampling Rate", ["1min", "6s"], index=0, key="prep_rate")
        
        st.header("1. Raw Data Generation")
        st.markdown("Simulating raw power readings (Mains) and ground truth (Appliance).")
        t, agg, app = generate_synthetic_data(length, window_size)
        
        fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
        ax_raw.plot(t, agg, label="Mains (Aggregate)", color='black', alpha=0.7)
        ax_raw.plot(t, app, label="Appliance (Target)", color='red', alpha=0.7)
        ax_raw.set_xlabel("Time Step")
        ax_raw.set_ylabel("Power (W)")
        ax_raw.legend()
        st.pyplot(fig_raw)
        
        st.info(f"Raw Data Shape: {agg.shape}")
        
        # Windowing logic for Synthetic
        st.header("2. Windowing & Reshaping")
        n_windows = length // window_size
        agg_windows = agg[:n_windows*window_size].reshape(n_windows, window_size)
        app_windows = app[:n_windows*window_size].reshape(n_windows, window_size)
        
        X_data = np.zeros((n_windows, 2, 2, window_size))
        X_data[:, 0, 0, :] = agg_windows 
        X_data[:, 1, 0, :] = app_windows 
        X_data[:, 1, 1, :] = (app_windows > 10).astype(float)
        
        st_date_df = pd.DataFrame({"start_date": pd.date_range(start="2024-01-01", periods=n_windows, freq=f"{window_size}min")})

    else: # Real Data
        dataset_name = st.sidebar.selectbox("Dataset", ["UKDALE", "REFIT"], key="prep_dataset")
        window_size = st.sidebar.number_input("Window Size", value=256, key="prep_win_real")
        sampling_rate = st.sidebar.selectbox("Sampling Rate", ["1min", "6s", "10s"], index=0, key="prep_rate_real")
        
        if dataset_name == "UKDALE":
            house_idx = st.sidebar.number_input("House Index", min_value=1, max_value=5, value=1, key="prep_house")
            appliance = st.sidebar.selectbox("Appliance", ["fridge", "kettle", "washing_machine", "dishwasher", "microwave"], key="prep_app")
        else: # REFIT
            house_idx = st.sidebar.number_input("House Index", min_value=1, max_value=21, value=1, key="prep_house")
            appliance = st.sidebar.selectbox("Appliance", ["Fridge", "Kettle", "WashingMachine", "Dishwasher", "Microwave"], key="prep_app")
            
        load_clicked = st.sidebar.button("Load Data", key="prep_load")
        
        if load_clicked or 'X_real' in st.session_state:
            # Load only if button clicked or already loaded
            if 'X_real' not in st.session_state or load_clicked:
                X, st_date = load_real_data(dataset_name, house_idx, appliance, window_size, sampling_rate)
                if X is not None:
                    st.session_state['X_real'] = X
                    st.session_state['st_date_real'] = st_date
                else:
                    st.stop()
            
            X_data = st.session_state['X_real']
            st_date_df = st.session_state['st_date_real']
            n_windows = X_data.shape[0]
            
            st.header("1. Loaded Real Data")
            st.success(f"Loaded {n_windows} windows from {dataset_name} House {house_idx} ({appliance})")
            st.write(f"Data Shape: `{X_data.shape}`") # [N, Apps, 2, Win]
            
            # Visualize a sample window
            win_idx_raw = st.slider("Select Raw Window Index", 0, n_windows-1, 0, key="prep_raw_idx")
            
            fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
            ax_raw.plot(X_data[win_idx_raw, 0, 0, :], label="Mains", color='black', alpha=0.7)
            ax_raw.plot(X_data[win_idx_raw, 1, 0, :], label=f"{appliance}", color='red', alpha=0.7)
            ax_raw.set_title(f"Window {win_idx_raw} (Raw)")
            ax_raw.legend()
            st.pyplot(fig_raw)
            
            st.header("2. Windowing & Reshaping")
            st.markdown("Real data is already windowed by the DataBuilder.")
        else:
            st.info("Click 'Load Data' to fetch real data from local storage.")
            st.stop()

    if 'X_data' in locals():
        st.write(f"Reshaped Data (X_data) Shape: `{X_data.shape}`")
    st.markdown("Dimensions: `[Batch, Appliances(Agg+App), Channels(Power/State), Time]`")
    
    n_windows = X_data.shape[0]
    
    # Display a specific window
    win_idx = st.slider("Select Window Index", 0, n_windows-1, 0, key="prep_win_idx")
    
    fig_win, ax_win = plt.subplots(figsize=(10, 3))
    ax_win.plot(X_data[win_idx, 0, 0, :], label="Window Mains", color='black')
    ax_win.plot(X_data[win_idx, 1, 0, :], label="Window Appliance", color='red')
    ax_win.set_title(f"Window {win_idx}")
    ax_win.legend()
    st.pyplot(fig_win)

    # 3. Scaling
    st.header("3. Scaling (NILMscaler)")
    st.markdown("Normalizing power values (e.g., dividing by max or standard scaling).")
    
    scaler = NILMscaler(power_scaling_type="MaxScaling", appliance_scaling_type="MaxScaling")
    scaler.fit(X_data)
    
    # Transform
    X_scaled = scaler.transform(X_data) # Note: transform returns a copy
    
    st.write(f"Max Power (Mains): {scaler.power_stat2:.2f}")
    st.write(f"Max Power (Appliance): {scaler.appliance_stat2[0]:.2f}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before Scaling")
        st.write(X_data[win_idx, 0, 0, :5]) # First 5 values
    with col2:
        st.subheader("After Scaling")
        st.write(X_scaled[win_idx, 0, 0, :5])

    fig_scaled, ax_scaled = plt.subplots(figsize=(10, 3))
    ax_scaled.plot(X_scaled[win_idx, 0, 0, :], label="Scaled Mains", color='blue')
    ax_scaled.plot(X_scaled[win_idx, 1, 0, :], label="Scaled Appliance", color='orange')
    ax_scaled.set_title(f"Window {win_idx} (Scaled)")
    ax_scaled.legend()
    st.pyplot(fig_scaled)

    # 4. Time Embedding Visualization
    st.header("4. Time Embedding Visualization")
    st.markdown("Visualizing how timestamps are converted into continuous features for the model.")
    
    col_time1, col_time2 = st.columns([1, 2])
    with col_time1:
        st.markdown("""
        **Why Time Embeddings?**
        Neural networks struggle with raw cyclic values (e.g., Hour 23 vs Hour 0 are far apart numerically but close in time).
        
        **Solution: Sin/Cos Encoding**
        We map time $t$ to two coordinates:
        - $sin(2 \pi t / Period)$
        - $cos(2 \pi t / Period)$
        
        This preserves cyclicity and continuity.
        """)
        
    with col_time2:
        # Demo Time Encoding
        demo_hours = np.arange(0, 24, 0.1)
        sin_h = np.sin(2 * np.pi * demo_hours / 24.0)
        cos_h = np.cos(2 * np.pi * demo_hours / 24.0)
        
        fig_time, ax_time = plt.subplots(figsize=(8, 3))
        ax_time.plot(demo_hours, sin_h, label="Sin(Hour)")
        ax_time.plot(demo_hours, cos_h, label="Cos(Hour)")
        ax_time.set_xlabel("Hour of Day")
        ax_time.set_ylabel("Encoding Value")
        ax_time.legend()
        ax_time.grid(True, alpha=0.3)
        st.pyplot(fig_time)

    # 5. Dataset (Model Input)
    st.header("5. Model Input (NILMDataset)")
    st.markdown("Final preparation: Adding exogenous variables (time info) and converting to Tensors.")
    
    # Instantiate Dataset
    dataset = NILMDataset(
        X_scaled,
        list_exo_variables=["hour", "dow", "month"],
        st_date=st_date_df,
        freq="1min", # 1 min
        mask_date="start_date"
    )
    
    sample = dataset[win_idx]
    
    # NILMDataset returns a tuple: (Input, Target Power, Target State)
    if isinstance(sample, tuple):
        model_input, target_power, target_state = sample
        st.write(f"Dataset Output Shapes:")
        st.write(f"- Model Input: `{model_input.shape}`")
        st.write(f"- Target Power: `{target_power.shape}`")
        st.write(f"- Target State: `{target_state.shape}`")
        
        # Visualize Input to Model (Channel 0 + Exo) vs Target (Appliance Power)
        # model_input: [Channels, Time] -> Channel 0 is usually Aggregate Power
        input_channels = model_input[0, :] 
        # target_power: [1, Time]
        target_channels = target_power[0, :]
    else:
        # Fallback if pretraining=True or other config
        model_input = sample
        st.write(f"Dataset Output Shape: `{sample.shape}`")
        input_channels = sample[0, :]
        target_channels = np.zeros_like(input_channels)

    st.write("Channels breakdown (Model Input):")
    st.write(f"- 0: Mains Power (Scaled)")
    if model_input.shape[0] > 1:
        st.write(f"- 1+: Exogenous Variables (Time encoding)")
    
    st.subheader("Final Tensor Visualization")
    fig_final, ax_final = plt.subplots(figsize=(10, 4))
    ax_final.plot(input_channels, label="Model Input (Mains)", linewidth=2)
    ax_final.plot(target_channels, label="Target (Appliance)", linestyle="--")
    ax_final.set_title("Data ready for Training/Inference")
    ax_final.legend()
    st.pyplot(fig_final)
    
    st.success("Preprocessing Pipeline Visualization Complete!")

def run_training_tab():
    st.header("Training Process Visualization")
    st.markdown("Watch the model learn step-by-step.")

    st.sidebar.header("Training Configuration")
    
    # Use Session State data if available, else generate synthetic
    if 'X_real' in st.session_state:
        st.info("Using loaded Real Data for training.")
        X_data = st.session_state['X_real']
        
        if 'st_date_real' in st.session_state:
            st_date_df = st.session_state['st_date_real']
        else:
            # Fallback
            st_date_df = pd.DataFrame({"start_date": pd.date_range(start="2024-01-01", periods=X_data.shape[0], freq=f"{X_data.shape[3]}min")})

        # Limit data for demo speed
        if X_data.shape[0] > 500:
            X_data = X_data[:500]
            st_date_df = st_date_df.iloc[:500].reset_index(drop=True)
            st.warning("Data truncated to 500 windows for faster demo.")
    else:
        st.info("Using Synthetic Data for training.")
        t, agg, app = generate_synthetic_data(length=5000, window_size=128)
        n_windows = 5000 // 128
        agg_windows = agg[:n_windows*128].reshape(n_windows, 128)
        app_windows = app[:n_windows*128].reshape(n_windows, 128)
        X_data = np.zeros((n_windows, 2, 2, 128))
        X_data[:, 0, 0, :] = agg_windows 
        X_data[:, 1, 0, :] = app_windows
        
        st_date_df = pd.DataFrame({"start_date": pd.date_range(start="2024-01-01", periods=n_windows, freq="128min")})
    
    # Scaling
    scaler = NILMscaler(power_scaling_type="MaxScaling", appliance_scaling_type="MaxScaling")
    scaler.fit(X_data)
    X_scaled = scaler.transform(X_data)
    
    # Prepare Dataset with Time Features
    dataset = NILMDataset(
        X_scaled, 
        list_exo_variables=["hour", "dow"], # Add time features
        freq="1min",
        st_date=st_date_df,
        mask_date="start_date"
    )
    
    # Hyperparameters
    epochs = st.sidebar.slider("Epochs", 5, 50, 10, key="train_epochs")
    lr = st.sidebar.select_slider("Learning Rate", options=[1e-4, 1e-3, 1e-2], value=1e-3, key="train_lr")
    
    start_train = st.button("Start Training Demo")
    
    if start_train:
        # Initialize Model
        window_size = X_data.shape[3]
        
        # Get a sample to determine channels
        sample_item = dataset[0]
        if isinstance(sample_item, tuple):
            sample_input = sample_item[0] # [Channels, Time]
            c_in = sample_input.shape[0]
        else:
            c_in = sample_item.shape[0]
            
        st.write(f"Model Input Channels (c_in): {c_in}")
        
        # Model Architecture Explanation
        st.markdown("""
        ### Model Architecture: UNetNiLM
        This model combines a **UNet-like Encoder-Decoder** with a **Transformer-style Encoder** and an **MLP Head**.
        
        1.  **UNet Block (CNN 1D)**: Extracts local temporal features using convolutional layers with skip connections. It captures high-frequency patterns (signatures) of the appliance.
        2.  **Encoder (Transformer/Attention)**: Processes the feature map to capture long-range dependencies and context.
        3.  **MLP Head**: A Multi-Layer Perceptron that projects the latent representation to the final output (power consumption).
        
        **Loss Function**: 
        - **SmoothL1Loss**: Robust regression loss (less sensitive to outliers than MSE).
        - **Negative Penalty**: Penalizes negative predictions (Energy cannot be negative).
        """)

        model = UNetNiLM(
            c_in=c_in, 
            window_size=window_size,
            num_layers=4,
            features_start=8,
            num_classes=1, 
            d_model=32
        )
        
        # Hooks for visualization
        activations = {}
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook

        # Register hooks
        model.unet_block.register_forward_hook(get_activation('1. UNet Block Output'))
        model.encoder.register_forward_hook(get_activation('2. Encoder Output'))
        model.mlp.register_forward_hook(get_activation('3. MLP Output'))
        
        # Training Setup matching CondiNILM logic
        criterion = nn.SmoothL1Loss()
        # Use AdamW as per config/trainer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
        
        # Split Data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        # Fix a validation sample for visualization
        val_sample = val_dataset[0] 
        # val_sample is tuple (input, target_p, target_s) or just input if simple.
        # Check structure:
        if isinstance(val_sample, tuple):
             val_input_tensor = torch.tensor(val_sample[0]).float().unsqueeze(0) # [1, C, T]
             val_target_tensor = torch.tensor(val_sample[1]).float().unsqueeze(0)
        else:
             val_input_tensor = torch.tensor(val_sample).float().unsqueeze(0)
             val_target_tensor = torch.zeros_like(val_input_tensor[:, 0:1, :]) # Dummy

        # Placeholders for live updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        col_metrics, col_visuals = st.columns([1, 2])
        
        with col_metrics:
            st.subheader("Loss History")
            chart_loss = st.line_chart([])
        
        with col_visuals:
            st.subheader("Live Layer-wise Visualization")
            plot_placeholder = st.empty()
        
        loss_history = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch in train_loader:
                # batch: [input, target_p, target_s]
                if isinstance(batch, list):
                    x = batch[0].float()
                    y = batch[1].float()
                else:
                    x = batch.float()
                    y = x[:, 0:1, :] # Dummy

                optimizer.zero_grad()
                pred = model(x) 
                
                # Custom Loss Logic
                loss_main = criterion(pred, y)
                neg_penalty = torch.relu(-pred).mean()
                loss = loss_main + 0.1 * neg_penalty
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            loss_history.append(avg_loss)
            chart_loss.line_chart(loss_history)
            
            # Validation & Visualization
            model.eval()
            with torch.no_grad():
                # Forward pass with hooks
                # Clear previous activations just in case
                activations = {} 
                # Re-registering is not needed, hooks persist. 
                # But we need to ensure we capture the *validation* pass.
                # The hooks capture every forward pass. We'll use the latest one.
                
                val_pred = model(val_input_tensor)
                
                # Visualization Logic
                fig = plt.figure(figsize=(15, 12))
                gs = fig.add_gridspec(5, 1) # Input, UNet, Encoder, MLP, Output
                
                # 1. Input Layers (Mains + Time)
                ax1 = fig.add_subplot(gs[0, 0])
                input_np = val_input_tensor.squeeze().numpy()
                # Normalize for vis if needed, but raw is fine
                ax1.plot(input_np[0, :], label="Mains Power (Scaled)", color='black')
                ax1.plot(input_np[1, :], label="Hour Info", color='orange', linestyle='--')
                if input_np.shape[0] > 2:
                    ax1.plot(input_np[2, :], label="Day of Week", color='green', linestyle=':')
                ax1.set_title("Input Data: Mains Power + Time Features (Positional Encoding)")
                ax1.legend(loc='upper right')
                
                # 2. UNet Block Output (Feature Map Heatmap)
                ax2 = fig.add_subplot(gs[1, 0])
                if '1. UNet Block Output' in activations:
                    # Shape: [1, Features, Time]
                    act = activations['1. UNet Block Output'].squeeze().numpy()
                    im2 = ax2.imshow(act, aspect='auto', cmap='viridis', interpolation='nearest')
                    ax2.set_title(f"Layer 1: UNet Block Output Features (Shape: {act.shape})")
                    ax2.set_ylabel("Channels")
                    plt.colorbar(im2, ax=ax2)
                
                # 3. Encoder Output
                ax3 = fig.add_subplot(gs[2, 0])
                if '2. Encoder Output' in activations:
                    act = activations['2. Encoder Output'].squeeze().numpy()
                    im3 = ax3.imshow(act, aspect='auto', cmap='magma', interpolation='nearest')
                    ax3.set_title(f"Layer 2: Encoder Output (Latent Representation) (Shape: {act.shape})")
                    ax3.set_ylabel("Channels")
                    plt.colorbar(im3, ax=ax3)
                    
                # 4. MLP Output (Vector)
                ax4 = fig.add_subplot(gs[3, 0])
                if '3. MLP Output' in activations:
                    # Shape: [1, 1024] -> Flattened
                    act = activations['3. MLP Output'].squeeze().numpy()
                    # Reshape for better vis if it's 1D
                    act_vis = act.reshape(1, -1)
                    im4 = ax4.imshow(act_vis, aspect='auto', cmap='plasma')
                    ax4.set_title(f"Layer 3: MLP Output (Global Context) (Shape: {act.shape})")
                    ax4.set_yticks([])
                
                # 5. Final Prediction vs Ground Truth
                ax5 = fig.add_subplot(gs[4, 0])
                val_pred_np = val_pred.squeeze().numpy()
                val_target_np = val_target_tensor.squeeze().numpy()
                
                # Rescale for display
                pred_real = val_pred_np * scaler.appliance_stat2[0]
                target_real = val_target_np * scaler.appliance_stat2[0]
                
                ax5.plot(target_real, label="Ground Truth", color='black', alpha=0.5)
                ax5.plot(pred_real, label=f"Prediction (Epoch {epoch+1})", color='red')
                ax5.set_title(f"Final Output: Prediction vs Truth (Epoch {epoch+1})")
                ax5.legend()
                
                plt.tight_layout()
                plot_placeholder.pyplot(fig)
                plt.close(fig)

            status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            progress_bar.progress((epoch + 1) / epochs)
            # time.sleep(0.1) 

        st.success("Training & Visualization Completed!")

def main():
    st.set_page_config(page_title="CondiNILM Visualization", layout="wide")
    st.title("CondiNILM: Pipeline & Training Visualization")
    
    tab1, tab2 = st.tabs(["Preprocessing Pipeline", "Training Demo"])
    
    with tab1:
        run_preprocessing_tab()
    
    with tab2:
        run_training_tab()

if __name__ == "__main__":
    main()
