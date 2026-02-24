"""NILM evaluation metrics: classification, regression, and energy aggregation."""

import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.helpers.preprocessing import create_exogene
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
)


EPS = 1e-12
BINARY_THRESHOLD = 0.5


class Classifmetrics:
    """Binary classification metrics: accuracy, precision, recall, F1, ROC-AUC, AP."""

    def __init__(self, round_to=5):
        self.round_to = round_to

    def __call__(self, y, y_hat):
        metrics = {}

        y = np.asarray(y)
        y_hat = np.asarray(y_hat)
        if y.size == 0 or y_hat.size == 0:
            return {
                "ACCURACY": 0.0,
                "BALANCED_ACCURACY": 0.0,
                "PRECISION": 0.0,
                "RECALL": 0.0,
                "F1_SCORE": 0.0,
                "F1_SCORE_MACRO": 0.0,
                "ROC_AUC_SCORE": 0.0,
                "AP": 0.0,
            }

        y_hat_round = y_hat.round()

        metrics["ACCURACY"] = round(accuracy_score(y, y_hat_round), self.round_to)
        metrics["BALANCED_ACCURACY"] = round(
            balanced_accuracy_score(y, y_hat_round), self.round_to
        )

        metrics["PRECISION"] = round(
            precision_score(y, y_hat_round, zero_division=0),
            self.round_to,
        )
        metrics["RECALL"] = round(
            recall_score(y, y_hat_round, zero_division=0),
            self.round_to,
        )
        metrics["F1_SCORE"] = round(
            f1_score(y, y_hat_round, average="binary", zero_division=0),
            self.round_to,
        )
        metrics["F1_SCORE_MACRO"] = round(
            f1_score(y, y_hat_round, average="macro", zero_division=0),
            self.round_to,
        )

        unique_classes = np.unique(y)
        if len(unique_classes) >= 2:
            metrics["ROC_AUC_SCORE"] = round(roc_auc_score(y, y_hat), self.round_to)
            metrics["AP"] = round(average_precision_score(y, y_hat), self.round_to)
        else:
            metrics["ROC_AUC_SCORE"] = 0.0
            metrics["AP"] = 0.0

        return metrics


class NILMmetrics:
    """NILM-specific metrics: MAE, MSE, RMSE, TECA, NDE, SAE, MR, plus event detection F1."""

    def __init__(self, round_to=3):
        self.round_to = round_to

    def __call__(self, y=None, y_hat=None, y_state=None, y_hat_state=None):
        metrics = {}

        if y is not None:
            assert y_hat is not None, (
                "Target y_hat not provided, please provide y_hat to compute regression metrics."
            )
            y = np.nan_to_num(
                np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            y_hat = np.nan_to_num(
                np.asarray(y_hat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )

            if y.size == 0 or y_hat.size == 0:
                metrics["MAE"] = 0.0
                metrics["MSE"] = 0.0
                metrics["RMSE"] = 0.0
                metrics["TECA"] = 0.0
                metrics["NDE"] = 0.0
                metrics["SAE"] = 0.0
                metrics["MR"] = 0.0
            else:
                metrics["MAE"] = round(mean_absolute_error(y, y_hat), self.round_to)
                metrics["MSE"] = round(mean_squared_error(y, y_hat), self.round_to)
                metrics["RMSE"] = round(
                    np.sqrt(mean_squared_error(y, y_hat)), self.round_to
                )

                abs_y_sum = float(np.sum(np.abs(y)))
                if abs_y_sum < EPS:
                    metrics["TECA"] = 1.0 if np.sum(np.abs(y_hat)) < EPS else 0.0
                else:
                    metrics["TECA"] = round(
                        1 - ((np.sum(np.abs(y_hat - y))) / (2 * abs_y_sum)),
                        self.round_to,
                    )

                y_sq_sum = float(np.sum(y**2))
                if y_sq_sum < EPS:
                    metrics["NDE"] = 0.0 if np.sum(y_hat**2) < EPS else float("inf")
                else:
                    metrics["NDE"] = round(
                        (np.sum((y_hat - y) ** 2)) / y_sq_sum, self.round_to
                    )

                y_sum = float(np.sum(y))
                abs_y_sum_for_sae = max(abs(y_sum), EPS)
                metrics["SAE"] = round(
                    np.abs(np.sum(y_hat) - y_sum) / abs_y_sum_for_sae, self.round_to
                )

                y_clipped = np.maximum(y, 0)
                y_hat_clipped = np.maximum(y_hat, 0)
                mr_denom = float(np.sum(np.maximum(y_hat_clipped, y_clipped)))
                if mr_denom < EPS:
                    metrics["MR"] = 1.0
                else:
                    metrics["MR"] = round(
                        np.sum(np.minimum(y_hat_clipped, y_clipped)) / mr_denom,
                        self.round_to,
                    )

        if y_state is not None:
            assert y_hat_state is not None, (
                "Target y_hat_state not provided, please pass y_hat_state to compute classification metrics."
            )
            y_state = np.nan_to_num(
                np.asarray(y_state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            y_hat_state = np.nan_to_num(
                np.asarray(y_hat_state, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            if y_state.size == 0 or y_hat_state.size == 0:
                metrics["ACCURACY"] = 0.0
                metrics["BALANCED_ACCURACY"] = 0.0
                metrics["PRECISION"] = 0.0
                metrics["RECALL"] = 0.0
                metrics["F1_SCORE"] = 0.0
                return metrics

            y_state = (y_state > BINARY_THRESHOLD).astype(np.int64, copy=False)
            y_hat_state = (y_hat_state > BINARY_THRESHOLD).astype(np.int64, copy=False)

            metrics["ACCURACY"] = round(
                accuracy_score(y_state, y_hat_state), self.round_to
            )

            uniq_true = np.unique(y_state)
            if uniq_true.size >= 2:
                metrics["BALANCED_ACCURACY"] = round(
                    balanced_accuracy_score(y_state, y_hat_state), self.round_to
                )
                metrics["PRECISION"] = round(
                    precision_score(y_state, y_hat_state, zero_division=0),
                    self.round_to,
                )
                metrics["RECALL"] = round(
                    recall_score(y_state, y_hat_state, zero_division=0),
                    self.round_to,
                )
                metrics["F1_SCORE"] = round(
                    f1_score(y_state, y_hat_state, average="binary", zero_division=0),
                    self.round_to,
                )
            else:
                metrics["BALANCED_ACCURACY"] = metrics["ACCURACY"]
                metrics["PRECISION"] = 0.0
                metrics["RECALL"] = 0.0
                metrics["F1_SCORE"] = 0.0

        return metrics


class REGmetrics:
    """Regression metrics: MAE, MSE, RMSE, MAPE."""

    def __init__(self, round_to=5):
        self.round_to = round_to

    def __call__(self, y, y_hat):
        metrics = {}

        y_arr = np.nan_to_num(
            np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        y_hat_arr = np.nan_to_num(
            np.asarray(y_hat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

        if y_arr.size == 0 or y_hat_arr.size == 0:
            return {
                "MAE": 0.0,
                "MSE": 0.0,
                "RMSE": 0.0,
                "MAPE": 0.0,
            }

        metrics["MAE"] = round(mean_absolute_error(y_arr, y_hat_arr), self.round_to)
        metrics["MSE"] = round(mean_squared_error(y_arr, y_hat_arr), self.round_to)
        metrics["RMSE"] = round(
            np.sqrt(mean_squared_error(y_arr, y_hat_arr)), self.round_to
        )

        denom = np.maximum(np.abs(y_arr), EPS)
        mape = float(np.mean(np.abs((y_arr - y_hat_arr) / denom)))
        if not np.isfinite(mape):
            mape = 0.0
        metrics["MAPE"] = round(mape, self.round_to)

        return metrics


def eval_win_energy_aggregation(
    input_data_test,
    input_st_date_test,
    model,
    device,
    scaler,
    metrics,
    window_size,
    freq,
    cosinbase=True,
    new_range=(-1, 1),
    mask_metric="test_metrics",
    list_exo_variables=None,
    threshold_small_values=0,
    use_temperature=False,
    log_dict=None,
):
    """Evaluate per-household energy disaggregation at daily, weekly, and monthly granularity.

    Runs model inference on each test window, inverse-scales predictions, then aggregates
    true and predicted appliance power per household at D/W/ME frequencies. Computes
    regression metrics on both absolute power and appliance-to-aggregate ratios.
    Results are stored in log_dict under keys like "{mask_metric}_{freq_agg}".

    Args:
        input_data_test: 4D array [N, M+1, 2, L] of scaled test data.
        input_st_date_test: DataFrame with house IDs as index and start dates.
        model: Trained PyTorch model returning appliance power predictions.
        device: torch.device for inference.
        scaler: NILMscaler instance for inverse transforms.
        metrics: Callable returning a dict of metric_name -> value.
        window_size: Number of time steps per window.
        freq: Pandas frequency string for the original sampling rate.
        cosinbase: Whether to use sin/cos encoding for exogenous features.
        new_range: Target range for linear normalization of exogenous features.
        mask_metric: Prefix for log_dict keys.
        list_exo_variables: List of time features to append (e.g. ["hour", "dow"]).
        threshold_small_values: Predictions below this value are zeroed out.
        use_temperature: Whether to include the temperature channel as input.
        log_dict: Mutable dict to store evaluation results; skipped if None.
    """
    data_test = input_data_test.copy()
    st_date_test = input_st_date_test.copy()

    st_date_test.index.name = "ID_PDL"
    st_date_test = st_date_test.reset_index()
    list_pdl_test = st_date_test["ID_PDL"].unique()

    for freq_agg in ["D", "W", "ME"]:
        all_dfs = []

        true_app_power = []
        pred_app_power = []
        true_ratio = []
        pred_ratio = []

        for pdl in tqdm(list_pdl_test, desc=mask_metric + "_" + freq_agg, leave=False):
            tmp_st_date_test = st_date_test.loc[st_date_test["ID_PDL"] == pdl]

            list_index = tmp_st_date_test.index

            list_date = []
            pdl_total_power = []
            pdl_true_app_power = []
            pdl_pred_app_power = []

            for k, val in enumerate(list(list_index)):
                if list_exo_variables is not None:
                    if use_temperature:
                        input_seq = torch.Tensor(
                            create_exogene(
                                data_test[val, 0, :2, :],
                                tmp_st_date_test.iloc[k, 1],
                                list_exo_variables=list_exo_variables,
                                freq=freq,
                                cosinbase=cosinbase,
                                new_range=new_range,
                            )
                        )
                    else:
                        input_seq = torch.Tensor(
                            create_exogene(
                                data_test[val, 0, 0, :],
                                tmp_st_date_test.iloc[k, 1],
                                list_exo_variables=list_exo_variables,
                                freq=freq,
                                cosinbase=cosinbase,
                                new_range=new_range,
                            )
                        )
                else:
                    if use_temperature:
                        input_seq = torch.Tensor(
                            np.expand_dims(data_test[val, 0, :2, :], axis=0)
                        )
                    else:
                        input_seq = torch.Tensor(
                            np.expand_dims(
                                np.expand_dims(data_test[val, 0, 0, :], axis=0), axis=0
                            )
                        )

                with torch.no_grad():
                    pred = model(input_seq.to(device))
                    pred = scaler.inverse_transform_appliance(pred)
                    pred[pred < threshold_small_values] = 0
                    pred = pred.cpu().numpy().flatten()

                inv_scale = scaler.inverse_transform(data_test[val, :, :, :])

                agg = inv_scale[0, 0, :]
                app = inv_scale[1, 0, :]
                full_dates = pd.date_range(
                    tmp_st_date_test.iloc[k, 1], periods=window_size, freq=freq
                )
                pred_len = int(len(pred))
                if pred_len <= 0:
                    continue
                if pred_len != int(window_size):
                    start_idx = max((int(window_size) - pred_len) // 2, 0)
                    end_idx = start_idx + pred_len
                    dates = full_dates[start_idx:end_idx]
                    agg = agg[start_idx:end_idx]
                    app = app[start_idx:end_idx]
                else:
                    dates = full_dates

                list_date.extend(list(dates))
                pdl_total_power.extend(list(agg))
                pdl_true_app_power.extend(list(app))
                pdl_pred_app_power.extend(list(pred))
            # Enforce equal lengths across parallel lists
            min_len = min(len(list_date), len(pdl_total_power), len(pdl_true_app_power), len(pdl_pred_app_power))
            if min_len <= 0:
                continue
            if len(list_date) != min_len:
                list_date = list_date[:min_len]
            if len(pdl_total_power) != min_len:
                pdl_total_power = pdl_total_power[:min_len]
            if len(pdl_true_app_power) != min_len:
                pdl_true_app_power = pdl_true_app_power[:min_len]
            if len(pdl_pred_app_power) != min_len:
                pdl_pred_app_power = pdl_pred_app_power[:min_len]

            df_inst = pd.DataFrame(
                {
                    "date": list_date,
                    "total_power": pdl_total_power,
                    "true_app_power": pdl_true_app_power,
                    "pred_app_power": pdl_pred_app_power,
                }
            )
            df_inst["date"] = pd.to_datetime(df_inst["date"])
            df_inst = df_inst.set_index("date")

            df_inst = df_inst.groupby(pd.Grouper(freq=freq_agg)).sum()

            true_app_power.extend(df_inst["true_app_power"].tolist())
            pred_app_power.extend(df_inst["pred_app_power"].tolist())

            total_power_arr = df_inst["total_power"].values
            true_app_arr = df_inst["true_app_power"].values
            pred_app_arr = df_inst["pred_app_power"].values

            df_inst["true_ratio"] = np.divide(
                true_app_arr,
                total_power_arr,
                out=np.zeros_like(true_app_arr, dtype=np.float64),
                where=total_power_arr > EPS,
            )
            df_inst["pred_ratio"] = np.divide(
                pred_app_arr,
                total_power_arr,
                out=np.zeros_like(pred_app_arr, dtype=np.float64),
                where=total_power_arr > EPS,
            )

            df_inst = df_inst.replace([np.inf, -np.inf], 0)
            df_inst = df_inst.fillna(value=0)

            true_ratio.extend(df_inst["true_ratio"].tolist())
            pred_ratio.extend(df_inst["pred_ratio"].tolist())

            all_dfs.append(df_inst)

        df = pd.concat(all_dfs, axis=0) if all_dfs else pd.DataFrame()

        if log_dict is not None:
            log_dict[mask_metric + "_" + freq_agg] = metrics(
                np.array(true_app_power), np.array(pred_app_power)
            )

        true_ratio = np.nan_to_num(
            np.array(true_ratio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        pred_ratio = np.nan_to_num(
            np.array(pred_ratio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if log_dict is not None:
            tmp_dict_ratio = metrics(true_ratio, pred_ratio)

            for name_m, values in tmp_dict_ratio.items():
                tmp_dict_ratio[name_m] = values * 100

            log_dict[mask_metric + "_ratio_" + freq_agg] = tmp_dict_ratio
            log_dict[mask_metric + "_ratio_" + freq_agg]["True_Ratio"] = (
                np.mean(np.array(true_ratio)) * 100
            )

    return
