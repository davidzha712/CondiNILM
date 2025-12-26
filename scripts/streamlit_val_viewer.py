import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def _find_npz_files(root_dir):
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.endswith("_val_streamlit.npz"):
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def _load_npz(path):
    data = np.load(path)
    agg = data["agg_power"]
    target = data["target_power"]
    pred = data["pred_power"]
    return agg, target, pred


def _build_dataframe(agg, target, pred, seq_idx, app_idx):
    agg_seq = agg[seq_idx, 0, :]
    target_seq = target[seq_idx, app_idx, :]
    pred_seq = pred[seq_idx, app_idx, :]
    t = np.arange(agg_seq.shape[-1])
    df = pd.DataFrame(
        {
            "t": t,
            "aggregate": agg_seq,
            "target": target_seq,
            "prediction": pred_seq,
        }
    )
    return df


def main():
    st.title("NILM 验证集功率曲线可视化")

    root_dir = st.text_input("结果根目录", value="result")
    files = _find_npz_files(root_dir)
    if not files:
        st.warning("未在指定目录下找到 *_val_streamlit.npz 文件。")
        return

    file_path = st.selectbox("选择一次实验", files)
    agg, target, pred = _load_npz(file_path)

    n_seq = agg.shape[0]
    n_app = target.shape[1]
    length = agg.shape[-1]

    st.write(f"序列数: {n_seq}，窗口长度: {length}，设备数: {n_app}")

    seq_idx = st.slider("选择序列索引", min_value=0, max_value=max(n_seq - 1, 0), value=0)

    app_idx = 0
    if n_app > 1:
        app_idx = st.slider(
            "选择设备索引", min_value=0, max_value=max(n_app - 1, 0), value=0
        )

    df = _build_dataframe(agg, target, pred, seq_idx, app_idx)

    show_agg = st.checkbox("显示总功率 (Aggregate)", value=True)
    show_target = st.checkbox("显示目标功率 (Target)", value=True)
    show_pred = st.checkbox("显示预测功率 (Prediction)", value=True)

    fig, ax = plt.subplots()
    if show_agg:
        ax.plot(df["t"], df["aggregate"], label="Aggregate", color="#7f7f7f")
    if show_target:
        ax.plot(df["t"], df["target"], label="Target", color="#2ca02c")
    if show_pred:
        ax.plot(df["t"], df["prediction"], label="Prediction", color="#ff7f0e")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Power")
    ax.legend()
    st.pyplot(fig)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="下载当前序列数据 (CSV)",
        data=csv_bytes,
        file_name=f"val_seq_{seq_idx}_app_{app_idx}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
