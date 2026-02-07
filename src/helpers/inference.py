#################################################################################################################
#
# Inference helpers extracted from src/helpers/expes.py
#
#################################################################################################################

import torch


def _crop_center_tensor(x, ratio):
    """
    Crop tensor to center region for seq2subseq inference.

    Args:
        x: Tensor of shape (..., L) where L is sequence length
        ratio: Ratio of center region to keep (0-1). E.g., 0.5 keeps middle 50%.

    Returns:
        Cropped tensor of shape (..., L*ratio)
    """
    if ratio >= 1.0:
        return x
    L = x.shape[-1]
    crop_len = int(L * ratio)
    crop_len = max(1, crop_len)
    start = (L - crop_len) // 2
    end = start + crop_len
    return x[..., start:end]


def create_sliding_windows(sequence, window_size, output_ratio):
    """
    Create overlapping sliding windows for seq2subseq inference.

    The stride is set to center_size (window_size * output_ratio) so that
    consecutive center regions are adjacent, enabling 1:1 resolution output.

    Args:
        sequence: Tensor of shape (C, T) where T is total sequence length
        window_size: Size of each window (e.g., 128)
        output_ratio: Ratio of center region (e.g., 0.5)

    Returns:
        windows: Tensor of shape (N, C, window_size) - N windows
        padding_info: Dict with padding information for stitching
    """
    C, T = sequence.shape
    center_size = int(window_size * output_ratio)
    stride = center_size  # Stride = center_size for adjacent centers
    margin = (window_size - center_size) // 2  # Edge margin

    # Pad sequence so every point gets a center prediction
    # Need margin padding on both ends
    pad_left = margin
    pad_right = margin + (stride - (T % stride)) % stride  # Align to stride
    padded = torch.nn.functional.pad(sequence, (pad_left, pad_right), mode='replicate')
    T_padded = padded.shape[-1]

    # Extract windows
    windows = []
    for start in range(0, T_padded - window_size + 1, stride):
        window = padded[:, start:start + window_size]
        windows.append(window)

    if not windows:
        # Sequence too short, use single padded window
        if T_padded < window_size:
            extra_pad = window_size - T_padded
            padded = torch.nn.functional.pad(padded, (0, extra_pad), mode='replicate')
        windows.append(padded[:, :window_size])

    windows = torch.stack(windows, dim=0)  # (N, C, window_size)

    padding_info = {
        'original_length': T,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'center_size': center_size,
        'margin': margin,
        'n_windows': len(windows),
    }
    return windows, padding_info


def stitch_center_predictions(predictions, padding_info):
    """
    Stitch center predictions from sliding windows to form full sequence.

    Args:
        predictions: Tensor of shape (N, C, window_size) - predictions for each window
        padding_info: Dict from create_sliding_windows

    Returns:
        stitched: Tensor of shape (C, T) - full sequence prediction with 1:1 resolution
    """
    center_size = padding_info['center_size']
    margin = padding_info['margin']
    original_length = padding_info['original_length']

    # Extract center from each window
    centers = []
    for i in range(predictions.shape[0]):
        window_pred = predictions[i]  # (C, window_size)
        center = window_pred[:, margin:margin + center_size]  # (C, center_size)
        centers.append(center)

    # Concatenate centers
    stitched_padded = torch.cat(centers, dim=-1)  # (C, N * center_size)

    # Remove padding to get original length
    stitched = stitched_padded[:, :original_length]

    return stitched


def inference_seq2subseq(model, sequence, window_size, output_ratio, device, batch_size=32):
    """
    Full sequence inference with seq2subseq sliding window approach.

    Achieves 1:1 resolution output while avoiding boundary effects.

    Args:
        model: The trained model
        sequence: Input tensor of shape (C_in, T) - full sequence
        window_size: Size of each window (e.g., 128)
        output_ratio: Ratio of center region (e.g., 0.5)
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        prediction: Tensor of shape (C_out, T) - 1:1 resolution prediction
    """
    model.eval()
    sequence = sequence.to(device)

    # Create sliding windows
    windows, padding_info = create_sliding_windows(sequence, window_size, output_ratio)
    n_windows = windows.shape[0]

    # Batch inference
    all_predictions = []
    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            batch = windows[i:i + batch_size].to(device)  # (B, C_in, window_size)
            pred = model(batch)  # (B, C_out, window_size)
            all_predictions.append(pred.cpu())

    predictions = torch.cat(all_predictions, dim=0)  # (N, C_out, window_size)

    # Stitch center predictions
    stitched = stitch_center_predictions(predictions, padding_info)

    return stitched.to(device)
