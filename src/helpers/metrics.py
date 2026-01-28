#################################################################################################################
#
# @author : Siyi Li (TU Braunschweig)
# @description : NILMFormer - Metrics

#
#################################################################################################################

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
    mean_absolute_percentage_error,
)


# ========================================= Metrics ========================================= #


class Classifmetrics:
    """
    Basics metrics for classification
    """

    def __init__(self, round_to=5):
        self.round_to = round_to

    def __call__(self, y, y_hat):
        metrics = {}

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
            f1_score(y, y_hat_round, zero_division=0),
            self.round_to,
        )
        metrics["F1_SCORE_MACRO"] = round(
            f1_score(y, y_hat_round, average="macro", zero_division=0),
            self.round_to,
        )

        metrics["ROC_AUC_SCORE"] = round(roc_auc_score(y, y_hat), self.round_to)
        metrics["AP"] = round(average_precision_score(y, y_hat), self.round_to)

        return metrics


class NILMmetrics:
    """
    Basics metrics for NILM
    """

    def __init__(self, round_to=3):
        self.round_to = round_to

    def __call__(self, y=None, y_hat=None, y_state=None, y_hat_state=None):
        metrics = {}
        eps = 1e-12

        # ======= Basic regression Metrics ======= #
        if y is not None:
            assert y_hat is not None, (
                "Target y_hat not provided, please provide y_hat to compute regression metrics."
            )
            y = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            y_hat = np.nan_to_num(
                y_hat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )

            # MAE, MSE and RMSE
            metrics["MAE"] = round(mean_absolute_error(y, y_hat), self.round_to)
            metrics["MSE"] = round(mean_squared_error(y, y_hat), self.round_to)
            metrics["RMSE"] = round(
                np.sqrt(mean_squared_error(y, y_hat)), self.round_to
            )

            # =======  NILM Metrics ======= #

            # Total Energy Correctly Assigned (TECA)
            abs_y_sum = float(np.sum(np.abs(y)))
            abs_y_sum = max(abs_y_sum, eps)
            metrics["TECA"] = round(
                1 - ((np.sum(np.abs(y_hat - y))) / (2 * abs_y_sum)),
                self.round_to,
            )
            # Normalized Disaggregation Error (NDE)
            y_sq_sum = float(np.sum(y**2))
            y_sq_sum = max(y_sq_sum, eps)
            metrics["NDE"] = round(
                (np.sum((y_hat - y) ** 2)) / y_sq_sum, self.round_to
            )
            # Signal Aggregate Error (SAE)
            y_sum = float(np.sum(y))
            y_sum = max(y_sum, eps)
            metrics["SAE"] = round(
                np.abs(np.sum(y_hat) - np.sum(y)) / y_sum, self.round_to
            )
            # Matching Rate
            mr_denom = float(np.sum(np.maximum(y_hat, y)))
            mr_denom = max(mr_denom, eps)
            metrics["MR"] = round(
                np.sum(np.minimum(y_hat, y)) / mr_denom,
                self.round_to,
            )

        # =======  Event Detection Metrics ======= #
        if y_state is not None:
            assert y_hat_state is not None, (
                "Target y_hat_state not provided, please pass y_hat_state to compute classification metrics."
            )
            y_state = np.nan_to_num(
                y_state.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )
            y_hat_state = np.nan_to_num(
                y_hat_state.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
            )

            # Accuracy and Balanced Accuracy
            metrics["ACCURACY"] = round(
                accuracy_score(y_state, y_hat_state), self.round_to
            )
            uniq_true = np.unique(y_state)
            uniq_pred = np.unique(y_hat_state)
            if uniq_true.size >= 2 and uniq_pred.size >= 2:
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
                    f1_score(y_state, y_hat_state, zero_division=0),
                    self.round_to,
                )
            else:
                metrics["BALANCED_ACCURACY"] = metrics["ACCURACY"]
                metrics["PRECISION"] = 0.0
                metrics["RECALL"] = 0.0
                metrics["F1_SCORE"] = 0.0

        return metrics


class REGmetrics:
    """
    Basics regrssion metrics
    """

    def __init__(self, round_to=5):
        self.round_to = round_to

    def __call__(self, y, y_hat):
        metrics = {}

        metrics["MAE"] = round(mean_absolute_error(y, y_hat), self.round_to)
        metrics["MSE"] = round(mean_squared_error(y, y_hat), self.round_to)
        metrics["RMSE"] = round(np.sqrt(mean_squared_error(y, y_hat)), self.round_to)
        y_arr = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        y_hat_arr = np.nan_to_num(
            np.asarray(y_hat, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        denom = np.maximum(np.abs(y_arr), 1e-12)
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
    data_test = input_data_test.copy()
    st_date_test = input_st_date_test.copy()

    st_date_test.index.name = "ID_PDL"
    st_date_test = st_date_test.reset_index()
    list_pdl_test = st_date_test["ID_PDL"].unique()

    for freq_agg in ["D", "W", "ME"]:
        df = pd.DataFrame()

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

                pred = model(input_seq.to(device))

                pred = scaler.inverse_transform_appliance(pred)

                pred[pred < threshold_small_values] = 0

                inv_scale = scaler.inverse_transform(data_test[val, :, :, :])

                agg = inv_scale[0, 0, :]
                app = inv_scale[1, 0, :]

                pred = pred.detach().cpu().numpy().flatten()

                list_date.extend(
                    list(
                        pd.date_range(
                            tmp_st_date_test.iloc[k, 1], periods=window_size, freq=freq
                        )
                    )
                )
                pdl_total_power.extend(list(agg))
                pdl_true_app_power.extend(list(app))
                pdl_pred_app_power.extend(list(pred))

            df_inst = pd.DataFrame(
                list(
                    zip(
                        list_date,
                        pdl_total_power,
                        pdl_true_app_power,
                        pdl_pred_app_power,
                    )
                ),
                columns=["date", "total_power", "true_app_power", "pred_app_power"],
            )
            df_inst["date"] = pd.to_datetime(df_inst["date"])
            df_inst = df_inst.set_index("date")

            df_inst = df_inst.groupby(pd.Grouper(freq=freq_agg)).sum()
            df_inst += (
                1  # Prevent total power or appliance power is 0 to calculate ratio
            )

            true_app_power.extend(df_inst["true_app_power"].tolist())
            pred_app_power.extend(df_inst["pred_app_power"].tolist())

            df_inst["true_ratio"] = df_inst["true_app_power"] / df_inst["total_power"]
            df_inst["pred_ratio"] = df_inst["pred_app_power"] / df_inst["total_power"]

            df_inst = df_inst.replace([np.inf, -np.inf], 0)
            df_inst = df_inst.fillna(value=0)

            true_ratio.extend(df_inst["true_ratio"].tolist())
            pred_ratio.extend(df_inst["pred_ratio"].tolist())

            df = df_inst if not df.size else pd.concat((df, df_inst), axis=0)

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
