"""Data loading, windowing, train/valid splitting, and dataset builders for UKDALE, REFIT, and REDD."""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def nilmdataset_to_tser(data):
    """Convert a 4D NILM array to time series extrinsic regression format.

    Args:
        data: np.ndarray of shape [N, M+1, 2, L] from any DataBuilder.

    Returns:
        X: np.ndarray of shape [N, L] containing aggregate power.
        y: np.ndarray of shape [N] with total appliance energy per window.
    """
    X = data[:, 0, 0, :]
    y = np.sum(data[:, 1, 0, :], axis=-1)

    return X, y


def split_train_valid_test(data, test_size=0.2, valid_size=0, nb_label_col=1):
    if isinstance(data, pd.core.frame.DataFrame):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                data.iloc[:, :-nb_label_col],
                data.iloc[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid, y_train_valid, test_size=valid_size, random_state=0
            )

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data.iloc[:, :-nb_label_col],
                data.iloc[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            return X_train, y_train, X_test, y_test

    elif isinstance(data, np.ndarray):
        if valid_size != 0:
            X_train_valid, X_test, y_train_valid, y_test = train_test_split(
                data[:, :-nb_label_col],
                data[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_train_valid, y_train_valid, test_size=valid_size, random_state=0
            )
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                data[:, :-nb_label_col],
                data[:, -nb_label_col:],
                test_size=test_size,
                random_state=0,
            )

            return X_train, y_train, X_test, y_test
    else:
        raise Exception("Please provide pandas Dataframe or numpy array object.")


def split_train_valid_test_pdl(
    df_data, test_size=0.2, valid_size=0, nb_label_col=1, seed=0, return_df=False
):
    """Split a DataFrame into train/valid/test by unique index IDs (e.g. household IDs).

    Args:
        df_data: DataFrame with a non-unique index identifying groups.
        test_size: Fraction of groups for the test set.
        valid_size: Fraction of remaining groups for the validation set (0 to skip).
        nb_label_col: Number of trailing columns treated as labels.
        seed: Random seed for reproducibility.
        return_df: If True return DataFrames; otherwise return numpy arrays.

    Returns:
        (X_train, y_train, [X_valid, y_valid,] X_test, y_test) as arrays or DataFrames.
    """

    np.random.seed(seed)
    list_pdl = np.array(df_data.index.unique())
    np.random.shuffle(list_pdl)
    pdl_train_valid = list_pdl[: int(len(list_pdl) * (1 - test_size))]
    pdl_test = list_pdl[int(len(list_pdl) * (1 - test_size)) :]
    np.random.shuffle(pdl_train_valid)
    pdl_train = pdl_train_valid[: int(len(pdl_train_valid) * (1 - valid_size))]

    df_train = df_data.loc[pdl_train, :].copy()
    df_test = df_data.loc[pdl_test, :].copy()

    df_train = df_train.sample(frac=1, random_state=seed)
    df_test = df_test.sample(frac=1, random_state=seed)

    if valid_size != 0:
        pdl_valid = pdl_train_valid[int(len(pdl_train_valid) * (1 - valid_size)) :]
        df_valid = df_data.loc[pdl_valid, :].copy()
        df_valid = df_valid.sample(frac=1, random_state=seed)

    if return_df:
        if valid_size != 0:
            return df_train, df_valid, df_test
        else:
            return df_train, df_test
    else:
        X_train = df_train.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
        y_train = df_train.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)
        X_test = df_test.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
        y_test = df_test.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)

        if valid_size != 0:
            X_valid = df_valid.iloc[:, :-nb_label_col].to_numpy().astype(np.float32)
            y_valid = df_valid.iloc[:, -nb_label_col:].to_numpy().astype(np.float32)

            return X_train, y_train, X_valid, y_valid, X_test, y_test
        else:
            return X_train, y_train, X_test, y_test


def split_train_test_pdl_nilmdataset(
    data,
    st_date,
    seed=0,
    nb_house_test=None,
    perc_house_test=None,
    nb_house_valid=None,
    perc_house_valid=None,
):
    assert nb_house_test is not None or perc_house_test is not None
    assert len(data) == len(st_date)
    assert isinstance(st_date, pd.DataFrame)

    np.random.seed(seed)

    if nb_house_valid is not None or perc_house_valid is not None:
        assert (nb_house_test is not None and nb_house_valid is not None) or (
            perc_house_test is not None and perc_house_valid is not None
        )

    if len(data.shape) > 2:
        tmp_shape = data.shape
        data = data.reshape(data.shape[0], -1)

    data = pd.concat([st_date.reset_index(), pd.DataFrame(data)], axis=1).set_index(
        "index"
    )
    list_pdl = np.array(data.index.unique())
    np.random.shuffle(list_pdl)

    if nb_house_test is None:
        nb_house_test = max(1, int(len(list_pdl) * perc_house_test))
        if perc_house_valid is not None and nb_house_valid is None:
            nb_house_valid = max(1, int(len(list_pdl) * perc_house_valid))

    if nb_house_valid is not None:
        assert len(list_pdl) > nb_house_test + nb_house_valid
    else:
        assert len(list_pdl) > nb_house_test

    pdl_test = list_pdl[:nb_house_test]

    if nb_house_valid is not None:
        pdl_valid = list_pdl[nb_house_test : nb_house_test + nb_house_valid]
        pdl_train = list_pdl[nb_house_test + nb_house_valid :]
    else:
        pdl_train = list_pdl[nb_house_test:]

    df_train = data.loc[pdl_train, :].copy()
    df_test = data.loc[pdl_test, :].copy()

    st_date_train = df_train.iloc[:, :1]
    data_train = df_train.iloc[:, 1:].values.reshape(
        (len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3])
    )
    st_date_test = df_test.iloc[:, :1]
    data_test = df_test.iloc[:, 1:].values.reshape(
        (len(df_test), tmp_shape[1], tmp_shape[2], tmp_shape[3])
    )

    if nb_house_valid is not None:
        df_valid = data.loc[pdl_valid, :].copy()
        st_date_valid = df_valid.iloc[:, :1]
        data_valid = df_valid.iloc[:, 1:].values.reshape(
            (len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3])
        )

        return (
            data_train,
            st_date_train,
            data_valid,
            st_date_valid,
            data_test,
            st_date_test,
        )
    else:
        return data_train, st_date_train, data_test, st_date_test


def split_train_test_nilmdataset(data, st_date, perc_house_test=0.2, seed=0):
    np.random.seed(seed)

    data_len = np.arange(len(data))
    np.random.shuffle(data_len)

    split_index = int(len(data_len) * (1 - perc_house_test))
    train_idx, test_idx = data_len[:split_index], data_len[split_index:]

    data_train, st_date_train = data[train_idx], st_date.iloc[train_idx]
    data_test, st_date_test = data[test_idx], st_date.iloc[test_idx]

    return data_train, st_date_train, data_test, st_date_test


def split_train_valid_timeblock_nilmdataset(
    data,
    st_date,
    perc_valid=0.2,
    window_size=None,
    window_stride=None,
):
    assert len(data) == len(st_date)
    assert isinstance(st_date, pd.DataFrame)
    assert 0 < perc_valid < 1
    assert window_size is not None
    assert window_stride is not None

    if len(data.shape) > 2:
        tmp_shape = data.shape
        flat_data = data.reshape(data.shape[0], -1)
    else:
        tmp_shape = None
        flat_data = data

    df = pd.concat([st_date.reset_index(), pd.DataFrame(flat_data)], axis=1).set_index(
        "index"
    )

    house_ids = df.index.unique()
    df_train_list = []
    df_valid_list = []

    gap_windows = int(np.ceil(float(window_size) / float(window_stride)))

    for house in house_ids:
        df_house = df.loc[house]
        if isinstance(df_house, pd.Series):
            df_house = df_house.to_frame().T

        df_house = df_house.sort_values(by=st_date.columns[0])
        n = len(df_house)
        if n <= 1:
            continue

        n_valid = max(1, int(n * perc_valid))
        n_train = n - n_valid - gap_windows

        if n_train <= 0:
            split_idx = int(n * (1 - perc_valid))
            df_train_list.append(df_house.iloc[:split_idx])
            df_valid_list.append(df_house.iloc[split_idx:])
            continue

        if n_train + gap_windows + n_valid > n:
            overflow = n_train + gap_windows + n_valid - n
            if overflow > 0:
                reduce_train = min(overflow, max(0, n_train - 1))
                n_train -= reduce_train
                overflow -= reduce_train
            if overflow > 0:
                n_valid = max(1, n_valid - overflow)
            if n_train <= 0 or n_valid <= 0 or n_train + gap_windows + n_valid > n:
                split_idx = int(n * (1 - perc_valid))
                df_train_list.append(df_house.iloc[:split_idx])
                df_valid_list.append(df_house.iloc[split_idx:])
                continue

        start_valid = n_train + gap_windows
        end_valid = start_valid + n_valid

        if start_valid >= n:
            split_idx = int(n * (1 - perc_valid))
            df_train_list.append(df_house.iloc[:split_idx])
            df_valid_list.append(df_house.iloc[split_idx:])
            continue

        end_valid = min(end_valid, n)

        df_train_list.append(df_house.iloc[:n_train])
        df_valid_list.append(df_house.iloc[start_valid:end_valid])

    if not df_train_list or not df_valid_list:
        data_train, st_date_train, data_valid, st_date_valid = split_train_test_nilmdataset(
            data,
            st_date,
            perc_house_test=perc_valid,
            seed=0,
        )
        return data_train, st_date_train, data_valid, st_date_valid

    df_train = pd.concat(df_train_list, axis=0)
    df_valid = pd.concat(df_valid_list, axis=0)

    st_cols = st_date.shape[1]
    st_date_train = df_train.iloc[:, :st_cols]
    st_date_valid = df_valid.iloc[:, :st_cols]

    data_train_flat = df_train.iloc[:, st_cols:].to_numpy()
    data_valid_flat = df_valid.iloc[:, st_cols:].to_numpy()

    if tmp_shape is not None:
        data_train = data_train_flat.reshape(
            (len(df_train), tmp_shape[1], tmp_shape[2], tmp_shape[3])
        )
        data_valid = data_valid_flat.reshape(
            (len(df_valid), tmp_shape[1], tmp_shape[2], tmp_shape[3])
        )
    else:
        data_train = data_train_flat
        data_valid = data_valid_flat

    return data_train, st_date_train, data_valid, st_date_valid


def normalize_exogene(x, xmin, xmax, newRange):
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)

    norm = (x - xmin) / (xmax - xmin)
    if newRange == (0, 1):
        return norm
    elif newRange != (0, 1):
        return norm * (newRange[1] - newRange[0]) + newRange[0]


def create_exogene(
    values, st_date, list_exo_variables, freq, cosinbase=True, new_range=(-1, 1)
):
    if cosinbase:
        n_var = 2 * len(list_exo_variables)
    else:
        n_var = len(list_exo_variables)

    np_extra = np.zeros(
        (1, n_var, len(values[-1]) if len(values.shape) > 1 else len(values))
    ).astype(np.float32)

    tmp = pd.date_range(
        start=st_date,
        periods=len(values[-1]) if len(values.shape) > 1 else len(values),
        freq=freq,
    )

    k = 0
    for exo_var in list_exo_variables:
        if exo_var == "month":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.month.values / 12.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.month.values / 12.0)
                k += 2
            else:
                np_extra[k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=12, newRange=new_range
                )
                k += 1
        elif exo_var == "dom":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.day.values / 31.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.day.values / 31.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=12, newRange=new_range
                )
                k += 1
        elif exo_var == "dow":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.dayofweek.values / 7.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.dayofweek.values / 7.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=1, xmax=7, newRange=new_range
                )
                k += 1
        elif exo_var == "hour":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.hour.values / 24.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.hour.values / 24.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.month.values, xmin=0, xmax=24, newRange=new_range
                )
                k += 1
        elif exo_var == "minute":
            if cosinbase:
                np_extra[0, k, :] = np.sin(2 * np.pi * tmp.minute.values / 60.0)
                np_extra[0, k + 1, :] = np.cos(2 * np.pi * tmp.minute.values / 60.0)
                k += 2
            else:
                np_extra[0, k, :] = normalize_exogene(
                    tmp.minute.values, xmin=0, xmax=60, newRange=new_range
                )
                k += 1
        else:
            raise ValueError(
                "Embedding unknown for these Data. Only 'month', 'dow', 'dom', 'hour', 'minute' supported, received {}".format(
                    exo_var
                )
            )

    if len(values.shape) == 1:
        values = np.expand_dims(np.expand_dims(values, axis=0), axis=0)
    elif len(values.shape) == 2:
        values = np.expand_dims(values, axis=0)

    values = np.concatenate((values, np_extra), axis=1)

    return values


APPLIANCE_ALIASES = {
    "fridge": ["freezer", "fridge_freezer", "fridge-freezer"],
    "freezer": ["fridge", "fridge_freezer", "fridge-freezer"],
    "washing_machine": ["washer_dryer", "washer", "clothes_washer"],
    "dishwasher": ["dish_washer"],
    "microwave": ["microwave_oven"],
}


class UKDALE_DataBuilder(object):
    """Loads UK-DALE .dat files, resamples, computes appliance activation status,
    and produces windowed 4D arrays [N, M+1, 2, L] for NILM training."""

    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        soft_label=False,
        use_status_from_kelly_paper=True,
        use_appliance_aliases=True,
        appliance_params=None,
    ):
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.soft_label = soft_label
        self.use_appliance_aliases = use_appliance_aliases
        self._external_appliance_params = appliance_params

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        self._check_appliance_names()
        self.mask_app = ["aggregate"] + self.mask_app

        self.cutoff = 6000
        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        if self.use_status_from_kelly_paper:
            # Thresholds in Watts; duration params in 10s steps (base resampling rate)
            self.appliance_param = {
                "kettle": {
                    "min_threshold": 2000,
                    "max_threshold": 3100,
                    "min_on_duration": 1,
                    "min_off_duration": 0,
                    "min_activation_time": 1,
                },
                "fridge": {
                    "min_threshold": 50,
                    "max_threshold": 300,
                    "min_on_duration": 6,
                    "min_off_duration": 1,
                    "min_activation_time": 1,
                },
                "washing_machine": {
                    "min_threshold": 20,
                    "max_threshold": 2500,
                    "min_on_duration": 180,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
                "microwave": {
                    "min_threshold": 200,
                    "max_threshold": 3000,
                    "min_on_duration": 1,
                    "min_off_duration": 3,
                    "min_activation_time": 1,
                },
                "dishwasher": {
                    "min_threshold": 10,
                    "max_threshold": 2500,
                    "min_on_duration": 180,
                    "min_off_duration": 180,
                    "min_activation_time": 12,
                },
            }
        else:
            self.appliance_param = {
                "kettle": {"min_threshold": 500, "max_threshold": 6000},
                "washing_machine": {"min_threshold": 300, "max_threshold": 3000},
                "dishwasher": {"min_threshold": 300, "max_threshold": 3000},
                "microwave": {"min_threshold": 200, "max_threshold": 6000},
                "fridge": {"min_threshold": 50, "max_threshold": 300},
            }

        if self._external_appliance_params is not None:
            for app_name, params in self._external_appliance_params.items():
                app_key = app_name.lower()
                if app_key not in self.appliance_param:
                    self.appliance_param[app_key] = {}
                self.appliance_param[app_key].update(params)

    def get_house_data(self, house_indicies):
        assert len(house_indicies) == 1, (
            "get_house_data() implemented to get data from 1 house only at a time."
        )

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """Build a binary classification dataset from NILM windows.

        Returns:
            X: np.ndarray [N, L] of aggregate power windows.
            y: np.ndarray [N] with 1 if appliance was active in the window, else 0.
            st_date: DataFrame with house IDs as index and 'start_date' column.
        """
        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies):
        """Build the windowed 4D NILM dataset from specified houses.

        Returns:
            data: np.ndarray [N, M+1, 2, L] where axis 1 index 0 is aggregate,
                  axis 2 index 0 is power and index 1 is activation status.
            st_date: DataFrame with house IDs as index and 'start_date' column.
        """
        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []

            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            X = np.empty(
                (len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size)
            )

            cpt = 0
            for i in range(n_wins):
                tmp = stems[
                    :,
                    i * self.window_stride : i * self.window_stride + self.window_size,
                ]

                if not self._check_anynan(tmp):
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2

                    cpt += 1

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for _ in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size
                else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size
                else tmp_st_date
            )

        return output_data, st_date

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status

    def _get_stems(self, dataframe):
        """Extract aggregate + per-appliance power and status into a 2D array.

        Returns:
            stems: np.ndarray [1 + 2*(M), T] with aggregate at row 0,
                   then (power, status) pairs for each appliance.
            dates: list of datetime index values from the dataframe.
        """
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = dataframe["aggregate"].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2

        return stems, list(dataframe.index)

    def _get_dataframe(self, indice):
        """Load one house's .dat files, merge channels, resample, and compute status."""
        candidates = [
            os.path.join(self.data_path, f"house_{indice}"),
            os.path.join(self.data_path, f"house{indice}"),
            os.path.join(self.data_path, f"House_{indice}"),
            os.path.join(self.data_path, f"House{indice}"),
        ]
        path_house = None
        for cand in candidates:
            if os.path.isdir(cand):
                path_house = cand + os.sep
                break
        if path_house is None:
            raise FileNotFoundError(
                f"No directory found for house index {indice} in {self.data_path}"
            )
        self._check_if_file_exist(path_house + "labels.dat")

        house_label = pd.read_csv(path_house + "labels.dat", sep=" ", header=None)
        house_label.columns = ["id", "appliance_name"]

        house_data = pd.read_csv(path_house + "channel_1.dat", sep=" ", header=None)
        house_data.columns = ["time", "aggregate"]
        house_data["time"] = pd.to_datetime(house_data["time"], unit="s")
        house_data = house_data.set_index("time")
        _base_rate = self.sampling_rate if self.sampling_rate in ("6s", "6S") else "10s"
        house_data = (
            house_data.resample(_base_rate).mean().ffill(limit=6)
        )
        house_data[house_data < 5] = 0

        if self.flag_week:
            tmp_min = house_data[
                (house_data.index.weekday == 1)
                & (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[
                (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]

        for appliance in self.mask_app[1:]:
            matched_name = None
            matched_id = None

            exact_match = house_label.loc[house_label["appliance_name"] == appliance]["id"].values
            if len(exact_match) != 0:
                matched_name = appliance
                matched_id = exact_match[0]
            elif self.use_appliance_aliases and appliance in APPLIANCE_ALIASES:
                for alias in APPLIANCE_ALIASES[appliance]:
                    alias_match = house_label.loc[house_label["appliance_name"] == alias]["id"].values
                    if len(alias_match) != 0:
                        matched_name = alias
                        matched_id = alias_match[0]
                        break
            
            if matched_id is not None:
                i = matched_id

                appl_data = pd.read_csv(
                    path_house + "channel_" + str(i) + ".dat", sep=" ", header=None
                )
                appl_data.columns = ["time", appliance]
                appl_data["time"] = pd.to_datetime(appl_data["time"], unit="s")
                appl_data = appl_data.set_index("time")
                appl_data = appl_data.resample(_base_rate).mean().ffill(limit=6)
                appl_data[appl_data < 5] = 0

                house_data = pd.merge(house_data, appl_data, how="inner", on="time")
                del appl_data
                house_data = house_data.clip(lower=0, upper=self.cutoff)
                house_data = house_data.sort_index()

                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                initial_status = (
                    (
                        (
                            house_data[appliance]
                            >= self.appliance_param[appliance]["min_threshold"]
                        )
                        & (
                            house_data[appliance]
                            <= self.appliance_param[appliance]["max_threshold"]
                        )
                    )
                    .astype(int)
                    .values
                )

                if self.use_status_from_kelly_paper:
                    house_data[appliance + "_status"] = self._compute_status(
                        initial_status,
                        self.appliance_param[appliance]["min_on_duration"],
                        self.appliance_param[appliance]["min_off_duration"],
                        self.appliance_param[appliance]["min_activation_time"],
                    )
                else:
                    house_data[appliance + "_status"] = initial_status

                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate not in ("10s", _base_rate):
            house_data = house_data.resample(self.sampling_rate).mean().ffill(limit=6)

        for appliance in self.mask_app[1:]:
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)
                else:
                    continue
            else:
                raise ValueError(
                    f"UKDALE house {indice}: appliance '{appliance}' not found in data. "
                    f"Available columns: {list(house_data.columns)}. "
                    f"Check dataset_params.yaml 'houses' list or datasets.yaml split config."
                )

        return house_data

    def _check_appliance_names(self):
        """Assert all requested appliance names are valid for UK-DALE."""
        for appliance in self.mask_app:
            assert appliance in [
                "washing_machine",
                "cooker",
                "dishwasher",
                "kettle",
                "fridge",
                "microwave",
                "electric_heater",
            ], f"Selected applicance unknow for UKDALE Dataset, got: {appliance}"

    def _check_if_file_exist(self, file):
        """Raise FileNotFoundError if path does not exist."""
        if not os.path.isfile(file):
            raise FileNotFoundError

    def _check_anynan(self, a):
        """Return True if the array contains any NaN."""
        return np.isnan(np.sum(a))


REFIT_APPLIANCE_ALIASES = {
    "Fridge": ["Fridge-Freezer", "Fridge & Freezer", "Fridge(garage)", "Freezer"],
    "WashingMachine": ["Washer Dryer", "WashingMachine (1)"],
    "Microwave": ["Combination Microwave"],
}


class REFIT_DataBuilder(object):
    """Loads REFIT CSV files, resamples, computes appliance activation status,
    and produces windowed 4D arrays [N, M+1, 2, L] for NILM training."""

    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        use_status_from_kelly_paper=True,
        soft_label=False,
        appliance_params=None,
    ):
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.soft_label = soft_label
        self._external_appliance_params = appliance_params

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period related (i.e. str type), got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        self._check_appliance_names()
        self.mask_app = ["Aggregate"] + self.mask_app

        self.cutoff = 10000
        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        if self.use_status_from_kelly_paper:
            # Thresholds in Watts; duration params in 10s steps
            self.appliance_param = {
                "Kettle": {
                    "min_threshold": 1000,
                    "max_threshold": 6000,
                    "min_on_duration": 1,
                    "min_off_duration": 0,
                    "min_activation_time": 0,
                },
                "WashingMachine": {
                    "min_threshold": 20,
                    "max_threshold": 3500,
                    "min_on_duration": 6,
                    "min_off_duration": 16,
                    "min_activation_time": 12,
                },
                "Dishwasher": {
                    "min_threshold": 50,
                    "max_threshold": 3000,
                    "min_on_duration": 2,
                    "min_off_duration": 180,
                    "min_activation_time": 12,
                },
                "Microwave": {
                    "min_threshold": 200,
                    "max_threshold": 6000,
                    "min_on_duration": 1,
                    "min_off_duration": 3,
                    "min_activation_time": 0,
                },
                "Fridge": {
                    "min_threshold": 50,
                    "max_threshold": 300,
                    "min_on_duration": 6,
                    "min_off_duration": 1,
                    "min_activation_time": 1,
                },
            }
        else:
            self.appliance_param = {
                "Kettle": {"min_threshold": 500, "max_threshold": 6000},
                "WashingMachine": {"min_threshold": 300, "max_threshold": 4000},
                "Dishwasher": {"min_threshold": 300, "max_threshold": 4000},
                "Microwave": {"min_threshold": 200, "max_threshold": 6000},
                "Fridge": {"min_threshold": 50, "max_threshold": 300},
            }

        if self._external_appliance_params is not None:
            for app_name, params in self._external_appliance_params.items():
                if app_name not in self.appliance_param:
                    self.appliance_param[app_name] = {}
                self.appliance_param[app_name].update(params)

    def get_house_data(self, house_indicies):
        assert len(house_indicies) == 1, (
            "get_house_data() implemented to get data from 1 house only at a time."
        )

        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """Build a binary classification dataset from NILM windows.

        Returns:
            X: np.ndarray [N, L] of aggregate power windows.
            y: np.ndarray [N] with 1 if appliance was active in the window, else 0.
            st_date: DataFrame with house IDs as index and 'start_date' column.
        """
        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))

        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1

        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies):
        """Build the windowed 4D NILM dataset from specified houses.

        Returns:
            data: np.ndarray [N, M+1, 2, L] where axis 1 index 0 is aggregate,
                  axis 2 index 0 is power and index 1 is activation status.
            st_date: DataFrame with house IDs as index and 'start_date' column.
        """
        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []

            data = self._get_dataframe(indice)
            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            X = np.empty(
                (len(house_indicies) * n_wins, len(self.mask_app), 2, self.window_size)
            )

            cpt = 0
            for i in range(n_wins):
                tmp = stems[
                    :,
                    i * self.window_stride : i * self.window_stride + self.window_size,
                ]

                if not self._check_anynan(tmp):
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])

                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)

                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2

                    cpt += 1

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for j in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size
                else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size
                else tmp_st_date
            )

        return output_data, st_date

    def _get_stems(self, dataframe):
        """Extract aggregate + per-appliance power and status into a 2D array."""
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = dataframe["Aggregate"].values

        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2

        return stems, list(dataframe.index)

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]
            assert len(on_events) == len(off_events)

        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1

        return tmp_status

    def _get_dataframe(self, indice):
        """Load one house's CSV, merge channels, resample, and compute activation status."""
        file = self.data_path + "CLEAN_House" + str(indice) + ".csv"
        self._check_if_file_exist(file)
        labels_houses = pd.read_csv(self.data_path + "HOUSES_Labels").set_index(
            "House_id"
        )

        house_data = pd.read_csv(file)
        house_data.columns = list(labels_houses.loc[int(indice)].values)

        # Resolve REFIT appliance name variants (e.g. "Fridge-Freezer" → "Fridge")
        for std_name, aliases in REFIT_APPLIANCE_ALIASES.items():
            if std_name in self.mask_app and std_name not in house_data.columns:
                for alias in aliases:
                    if alias in house_data.columns:
                        house_data = house_data.rename(columns={alias: std_name})
                        break

        house_data = house_data.set_index("Time").sort_index()
        house_data.index = pd.to_datetime(house_data.index)
        idx_to_drop = house_data[house_data["Issues"] == 1].index
        house_data = house_data.drop(index=idx_to_drop, axis=0)
        house_data = house_data.resample(rule="10s").mean().ffill(limit=9)
        house_data[house_data < 5] = 0
        house_data = house_data.clip(lower=0, upper=self.cutoff)
        house_data = house_data.sort_index()

        if self.flag_week:
            tmp_min = house_data[
                (house_data.index.weekday == 1)
                & (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[
                (house_data.index.hour == 0)
                & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            house_data = house_data[house_data.index >= tmp_min.index[0]]

        for appliance in self.mask_app[1:]:
            if appliance in house_data:
                # Temporarily replace NaN with -1 so threshold comparison ignores missing values
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                initial_status = (
                    (
                        (
                            house_data[appliance]
                            >= self.appliance_param[appliance]["min_threshold"]
                        )
                        & (
                            house_data[appliance]
                            <= self.appliance_param[appliance]["max_threshold"]
                        )
                    )
                    .astype(int)
                    .values
                )

                if self.use_status_from_kelly_paper:
                    house_data[appliance + "_status"] = self._compute_status(
                        initial_status,
                        self.appliance_param[appliance]["min_on_duration"],
                        self.appliance_param[appliance]["min_off_duration"],
                        self.appliance_param[appliance]["min_activation_time"],
                    )
                else:
                    house_data[appliance + "_status"] = initial_status

                house_data[appliance] = house_data[appliance].replace(-1, np.nan)

        if self.sampling_rate != "10s":
            house_data = house_data.resample(self.sampling_rate).mean().ffill(limit=6)

        tmp_list = ["Aggregate"]
        for appliance in self.mask_app[1:]:
            tmp_list.append(appliance)
            tmp_list.append(appliance + "_status")
            if appliance in house_data:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)
                else:
                    continue
            else:
                raise ValueError(
                    f"REFIT house {indice}: appliance '{appliance}' not found in data "
                    f"(even after alias resolution). "
                    f"Available columns: {list(house_data.columns)}. "
                    f"Check dataset_params.yaml 'houses' list or datasets.yaml split config."
                )

        house_data = house_data[tmp_list]

        return house_data

    def _check_appliance_names(self):
        """Assert all requested appliance names are valid for REFIT."""
        for appliance in self.mask_app:
            assert appliance in [
                "WashingMachine",
                "Dishwasher",
                "Kettle",
                "Microwave",
                "Fridge",
            ], f"Selected applicance unknow for REFIT Dataset, got: {appliance}"
        return

    def _check_if_file_exist(self, file):
        """Raise FileNotFoundError if path does not exist."""
        if not os.path.isfile(file):
            raise FileNotFoundError

    def _check_anynan(self, a):
        """Return True if the array contains any NaN."""
        return np.isnan(np.sum(a))



REDD_APPLIANCE_MAPPING = {
    "dish washer": "dishwasher",
    "washer dryer": "washing_machine",
    "electric space heater": "electric_heater",
    "electric stove": "cooker",
    "fridge": "fridge",
    "microwave": "microwave",
    "electric furnace": "electric_furnace",
    "CE appliance": "ce_appliance",
    "waste disposal unit": "waste_disposal",
}

REDD_APPLIANCE_REVERSE_MAPPING = {v: k for k, v in REDD_APPLIANCE_MAPPING.items()}

REDD_APPLIANCE_ALIASES = {
    "fridge": ["freezer", "fridge_freezer"],
    "washing_machine": ["washer_dryer", "washer dryer"],
    "dishwasher": ["dish_washer", "dish washer"],
    "microwave": ["microwave_oven"],
    "cooker": ["electric_stove", "electric stove", "stove"],
    "electric_heater": ["electric_space_heater", "electric space heater", "heater"],
}


class REDD_DataBuilder(object):
    """
    DataBuilder for REDD (Reference Energy Disaggregation Dataset).

    REDD data format (preprocessed CSV):
    - Multiple CSV files per house (e.g., redd_house1_0.csv to redd_house1_10.csv)
    - Columns: device names (with spaces) and 'main' (aggregate)
    - Each row represents a time step

    Output format matches UKDALE_DataBuilder:
    - 4D array: [N_sequences, M_appliances, 2, Window_Size]
      - Dimension 0: number of windows
      - Dimension 1: appliances (index 0 = aggregate, others = target appliances)
      - Dimension 2: 0 = power, 1 = status
      - Dimension 3: time steps within window
    """

    def __init__(
        self,
        data_path,
        mask_app,
        sampling_rate,
        window_size,
        window_stride=None,
        soft_label=False,
        use_status_from_kelly_paper=True,
        use_appliance_aliases=True,
        appliance_params=None,
    ):
        self.data_path = data_path
        self.mask_app = mask_app
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.soft_label = soft_label
        self.use_appliance_aliases = use_appliance_aliases
        self._external_appliance_params = appliance_params

        if isinstance(self.mask_app, str):
            self.mask_app = [self.mask_app]

        if isinstance(window_size, str):
            if window_size == "week":
                self.flag_week = True
                self.flag_day = False
                if (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 10080
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 1008
                else:
                    raise ValueError(
                        f"Only sampling rate 1min and 10min supported for window size='week', got: {sampling_rate}"
                    )
            elif window_size == "day":
                self.flag_week = False
                self.flag_day = True
                if self.sampling_rate == "30s":
                    self.window_size = 2880
                elif (self.sampling_rate == "1min") or (self.sampling_rate == "1T"):
                    self.window_size = 1440
                elif (self.sampling_rate == "10min") or (self.sampling_rate == "10T"):
                    self.window_size = 144
                else:
                    raise ValueError(
                        f"Only sampling rate 30s, 1min and 10min supported for window size='day', got: {sampling_rate}"
                    )
            else:
                raise ValueError(
                    f'Only window size = "day" or "week" for window period, got: {window_size}'
                )
        else:
            self.flag_week = False
            self.flag_day = False
            self.window_size = window_size

        if window_stride is not None:
            self.window_stride = window_stride
        else:
            self.window_stride = self.window_size

        self._check_appliance_names()
        self.mask_app = ["aggregate"] + self.mask_app

        self.cutoff = 6000
        self.use_status_from_kelly_paper = use_status_from_kelly_paper

        if self.use_status_from_kelly_paper:
            self.appliance_param = {
                "kettle": {
                    "min_threshold": 2000, "max_threshold": 3100,
                    "min_on_duration": 1, "min_off_duration": 0, "min_activation_time": 1,
                },
                "fridge": {
                    "min_threshold": 50, "max_threshold": 300,
                    "min_on_duration": 6, "min_off_duration": 1, "min_activation_time": 1,
                },
                "washing_machine": {
                    "min_threshold": 20, "max_threshold": 2500,
                    "min_on_duration": 180, "min_off_duration": 16, "min_activation_time": 12,
                },
                "microwave": {
                    "min_threshold": 200, "max_threshold": 3000,
                    "min_on_duration": 1, "min_off_duration": 3, "min_activation_time": 1,
                },
                "dishwasher": {
                    "min_threshold": 10, "max_threshold": 2500,
                    "min_on_duration": 180, "min_off_duration": 180, "min_activation_time": 12,
                },
                "cooker": {
                    "min_threshold": 100, "max_threshold": 5000,
                    "min_on_duration": 6, "min_off_duration": 3, "min_activation_time": 3,
                },
                "electric_heater": {
                    "min_threshold": 100, "max_threshold": 3000,
                    "min_on_duration": 30, "min_off_duration": 10, "min_activation_time": 6,
                },
                "electric_furnace": {
                    "min_threshold": 100, "max_threshold": 5000,
                    "min_on_duration": 60, "min_off_duration": 30, "min_activation_time": 12,
                },
                "ce_appliance": {
                    "min_threshold": 50, "max_threshold": 500,
                    "min_on_duration": 6, "min_off_duration": 3, "min_activation_time": 1,
                },
                "waste_disposal": {
                    "min_threshold": 100, "max_threshold": 1500,
                    "min_on_duration": 1, "min_off_duration": 1, "min_activation_time": 1,
                },
            }
        else:
            self.appliance_param = {
                "kettle": {"min_threshold": 500, "max_threshold": 6000},
                "washing_machine": {"min_threshold": 300, "max_threshold": 3000},
                "dishwasher": {"min_threshold": 300, "max_threshold": 3000},
                "microwave": {"min_threshold": 200, "max_threshold": 6000},
                "fridge": {"min_threshold": 50, "max_threshold": 300},
                "cooker": {"min_threshold": 100, "max_threshold": 5000},
                "electric_heater": {"min_threshold": 100, "max_threshold": 3000},
                "electric_furnace": {"min_threshold": 100, "max_threshold": 5000},
                "ce_appliance": {"min_threshold": 50, "max_threshold": 500},
                "waste_disposal": {"min_threshold": 100, "max_threshold": 1500},
            }

        if self._external_appliance_params is not None:
            for app_name, params in self._external_appliance_params.items():
                app_key = app_name.lower()
                if app_key not in self.appliance_param:
                    self.appliance_param[app_key] = {}
                self.appliance_param[app_key].update(params)

    def get_house_data(self, house_indicies):
        """Get raw data from a single house."""
        assert len(house_indicies) == 1, "get_house_data() for 1 house only at a time."
        return self._get_dataframe(house_indicies[0])

    def get_classif_dataset(self, house_indicies):
        """Process data to build classification dataset."""
        nilm_dataset, st_date = self.get_nilm_dataset(house_indicies)
        y = np.zeros(len(nilm_dataset))
        for idx in range(len(nilm_dataset)):
            if (nilm_dataset[idx, 1, 1, :] > 0).any():
                y[idx] = 1
        return nilm_dataset[:, 0, 0, :], y, st_date

    def get_nilm_dataset(self, house_indicies, auto_filter_devices=True):
        """
        Process data to build NILM dataset.

        Args:
            house_indicies: List of house numbers to process
            auto_filter_devices: If True, automatically filter devices based on availability

        Returns:
            - np.ndarray [N_ts, M_appliances, 2, Win_Size]
            - pandas.DataFrame with start indices
        """
        import logging

        if auto_filter_devices and len(self.mask_app) > 1:
            try:
                valid_devices = self.validate_devices_for_houses(
                    house_indicies, auto_filter=True, activity_threshold=0.3
                )
                original_devices = [d for d in self.mask_app if d != "aggregate"]
                if set(valid_devices) != set(original_devices):
                    self.mask_app = ["aggregate"] + valid_devices
                    logging.warning(
                        f"REDD: Auto-filtered devices from {original_devices} to {valid_devices} "
                        f"for houses {house_indicies}"
                    )
            except ValueError as e:
                logging.error(f"Device validation failed: {e}")
                raise

        output_data = np.array([])
        st_date = pd.DataFrame()

        for indice in house_indicies:
            tmp_list_st_date = []
            data = self._get_dataframe(indice)
            if data is None or len(data) == 0:
                continue

            stems, st_date_stems = self._get_stems(data)

            if self.window_size == self.window_stride:
                n_wins = len(data) // self.window_stride
            else:
                n_wins = 1 + ((len(data) - self.window_size) // self.window_stride)

            if n_wins <= 0:
                continue

            X = np.empty((n_wins, len(self.mask_app), 2, self.window_size))

            cpt = 0
            for i in range(n_wins):
                tmp = stems[:, i * self.window_stride : i * self.window_stride + self.window_size]
                if not self._check_anynan(tmp):
                    tmp_list_st_date.append(st_date_stems[i * self.window_stride])
                    X[cpt, 0, 0, :] = tmp[0, :]
                    X[cpt, 0, 1, :] = (tmp[0, :] > 0).astype(dtype=int)
                    key = 1
                    for j in range(1, len(self.mask_app)):
                        X[cpt, j, 0, :] = tmp[key, :]
                        X[cpt, j, 1, :] = tmp[key + 1, :]
                        key += 2
                    cpt += 1

            if cpt == 0:
                continue

            tmp_st_date = pd.DataFrame(
                data=tmp_list_st_date,
                index=[indice for _ in range(cpt)],
                columns=["start_date"],
            )
            output_data = (
                np.concatenate((output_data, X[:cpt, :, :, :]), axis=0)
                if output_data.size else X[:cpt, :, :, :]
            )
            st_date = (
                pd.concat([st_date, tmp_st_date], axis=0)
                if st_date.size else tmp_st_date
            )

        return output_data, st_date

    def _compute_status(self, initial_status, min_on, min_off, min_activation_time):
        """Compute appliance status with duration filtering."""
        tmp_status = np.zeros_like(initial_status)
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()
        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)
        if initial_status[-1]:
            events_idx = np.insert(events_idx, events_idx.size, initial_status.size)
        if events_idx.size == 0:
            return tmp_status

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off]
            off_events = off_events[np.roll(off_duration, -1) > min_off]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on]
            off_events = off_events[on_duration >= min_on]

        activation_durations = off_events - on_events
        valid_activations = activation_durations >= min_activation_time
        on_events = on_events[valid_activations]
        off_events = off_events[valid_activations]

        for on, off in zip(on_events, off_events):
            tmp_status[on:off] = 1
        return tmp_status

    def _get_stems(self, dataframe):
        """Extract power curves for each appliance."""
        stems = np.empty((1 + (len(self.mask_app) - 1) * 2, dataframe.shape[0]))
        stems[0, :] = dataframe["aggregate"].values
        key = 1
        for appliance in self.mask_app[1:]:
            stems[key, :] = dataframe[appliance].values
            stems[key + 1, :] = dataframe[appliance + "_status"].values
            key += 2
        return stems, list(dataframe.index)

    def _get_dataframe(self, indice):
        """Load REDD house data from CSV files."""
        import glob

        pattern = os.path.join(self.data_path, f"redd_house{indice}_*.csv")
        csv_files = sorted(glob.glob(pattern))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files for REDD house {indice} in {self.data_path}")

        all_data = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, index_col=0)
            all_data.append(df)

        house_data = pd.concat(all_data, ignore_index=True)

        if 'main' in house_data.columns:
            house_data = house_data.rename(columns={'main': 'aggregate'})

        column_mapping = {}
        for col in house_data.columns:
            if col != 'aggregate' and col in REDD_APPLIANCE_MAPPING:
                column_mapping[col] = REDD_APPLIANCE_MAPPING[col]
        if column_mapping:
            house_data = house_data.rename(columns=column_mapping)

        # Synthetic 1s time index (REDD CSV lacks timestamps)
        house_data.index = pd.date_range(start='2011-04-01', periods=len(house_data), freq='1s')
        house_data[house_data < 5] = 0

        for appliance in self.mask_app[1:]:
            matched_col = None

            if appliance in house_data.columns:
                matched_col = appliance
            elif self.use_appliance_aliases and appliance in REDD_APPLIANCE_ALIASES:
                for alias in REDD_APPLIANCE_ALIASES[appliance]:
                    if alias in house_data.columns:
                        house_data = house_data.rename(columns={alias: appliance})
                        matched_col = appliance
                        break
            if matched_col is None and appliance in REDD_APPLIANCE_REVERSE_MAPPING:
                redd_name = REDD_APPLIANCE_REVERSE_MAPPING[appliance]
                if redd_name in house_data.columns:
                    house_data = house_data.rename(columns={redd_name: appliance})
                    matched_col = appliance

            if matched_col is not None:
                house_data[appliance] = house_data[appliance].clip(lower=0, upper=self.cutoff)
                house_data[appliance] = house_data[appliance].replace(np.nan, -1)

                if appliance in self.appliance_param:
                    initial_status = (
                        (house_data[appliance] >= self.appliance_param[appliance]["min_threshold"])
                        & (house_data[appliance] <= self.appliance_param[appliance]["max_threshold"])
                    ).astype(int).values

                    if self.use_status_from_kelly_paper:
                        house_data[appliance + "_status"] = self._compute_status(
                            initial_status,
                            self.appliance_param[appliance]["min_on_duration"],
                            self.appliance_param[appliance]["min_off_duration"],
                            self.appliance_param[appliance]["min_activation_time"],
                        )
                    else:
                        house_data[appliance + "_status"] = initial_status
                else:
                    house_data[appliance + "_status"] = (house_data[appliance] > 5).astype(int)

                house_data[appliance] = house_data[appliance].replace(-1, np.nan)
            else:
                house_data[appliance] = 0
                house_data[appliance + "_status"] = 0

        if self.sampling_rate != "1s":
            house_data = house_data.resample(self.sampling_rate).mean()

        for appliance in self.mask_app[1:]:
            if appliance in house_data.columns:
                if not self.soft_label:
                    house_data[appliance + "_status"] = (
                        house_data[appliance + "_status"] > 0
                    ).astype(int)

        if self.flag_week:
            tmp_min = house_data[
                (house_data.index.weekday == 1) & (house_data.index.hour == 0)
                & (house_data.index.minute == 0) & (house_data.index.second == 0)
            ]
            if len(tmp_min) > 0:
                house_data = house_data[house_data.index >= tmp_min.index[0]]
        elif self.flag_day:
            tmp_min = house_data[
                (house_data.index.hour == 0) & (house_data.index.minute == 0)
                & (house_data.index.second == 0)
            ]
            if len(tmp_min) > 0:
                house_data = house_data[house_data.index >= tmp_min.index[0]]

        return house_data

    def _check_appliance_names(self):
        """Check that appliance names are valid for REDD dataset."""
        valid_appliances = [
            "dishwasher", "washing_machine", "fridge", "microwave", "cooker",
            "electric_heater", "electric_furnace", "ce_appliance", "waste_disposal", "kettle",
        ]
        for appliance in self.mask_app:
            if appliance not in valid_appliances:
                raise ValueError(f"Unknown appliance for REDD: {appliance}. Valid: {valid_appliances}")

    def _check_anynan(self, a):
        """Fast check of NaN in a numpy array."""
        return np.isnan(np.sum(a))

    @staticmethod
    def get_device_availability(data_path, houses=None, activity_threshold=0.3):
        """
        Analyze device availability across REDD houses.

        Args:
            data_path: Path to REDD data directory
            houses: List of house numbers to check (default: 1-6)
            activity_threshold: Minimum activity percentage to consider device "available"

        Returns:
            dict: {
                "per_house": {house_num: {device: activity_rate}},
                "common_devices": {(house_tuple): [devices]}
            }
        """
        import glob
        from itertools import combinations

        if houses is None:
            houses = [1, 2, 3, 4, 5, 6]

        house_devices = {}  # {house: {device: activity_rate}}

        for house in houses:
            pattern = os.path.join(data_path, f"redd_house{house}_*.csv")
            csv_files = sorted(glob.glob(pattern))

            if not csv_files:
                continue

            all_data = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file, index_col=0)
                all_data.append(df)
            house_data = pd.concat(all_data, ignore_index=True)

            devices = {}
            for col in house_data.columns:
                if col in ["main", "aggregate", "Unnamed: 0", "index"]:
                    continue
                std_name = REDD_APPLIANCE_MAPPING.get(col, col)
                if col in house_data.columns:
                    activity = (house_data[col] > 10).mean() * 100
                    devices[std_name] = activity

            house_devices[house] = devices

        common_devices = {}
        house_list = sorted(house_devices.keys())

        for n in range(len(house_list), 0, -1):
            for combo in combinations(house_list, n):
                device_sets = []
                for h in combo:
                    active_devices = {
                        d for d, act in house_devices.get(h, {}).items()
                        if act >= activity_threshold
                    }
                    device_sets.append(active_devices)

                if device_sets:
                    common = set.intersection(*device_sets)
                    if common:
                        common_devices[combo] = sorted(common)

        return {
            "per_house": house_devices,
            "common_devices": common_devices,
        }

    def validate_devices_for_houses(self, house_indicies, auto_filter=True, activity_threshold=0.3):
        """
        Validate that requested devices exist in the specified houses.

        Args:
            house_indicies: List of house numbers
            auto_filter: If True, filter out devices that don't exist; if False, raise error
            activity_threshold: Minimum activity percentage (default 0.3%)

        Returns:
            list: Valid device names (excluding 'aggregate')
        """
        availability = self.get_device_availability(
            self.data_path, houses=house_indicies, activity_threshold=activity_threshold
        )

        requested_devices = [d for d in self.mask_app if d != "aggregate"]
        valid_devices = []
        invalid_devices = []

        for device in requested_devices:
            device_valid = True
            for house in house_indicies:
                house_data = availability["per_house"].get(house, {})
                activity = house_data.get(device, 0)
                if activity < activity_threshold:
                    device_valid = False
                    break

            if device_valid:
                valid_devices.append(device)
            else:
                invalid_devices.append(device)

        if invalid_devices:
            msg = f"Devices {invalid_devices} have insufficient data (<{activity_threshold}%) in houses {house_indicies}"
            if auto_filter:
                import logging
                logging.warning(f"AUTO-FILTER: {msg}. Using only: {valid_devices}")
            else:
                raise ValueError(msg)

        if not valid_devices:
            raise ValueError(
                f"No valid devices found for houses {house_indicies}. "
                f"Requested: {requested_devices}. Check device availability with get_device_availability()."
            )

        return valid_devices

    @staticmethod
    def recommend_configuration(data_path, min_devices=2, min_houses=2, activity_threshold=0.3):
        """
        Recommend the best device/house configuration for multi-device training.

        Args:
            data_path: Path to REDD data
            min_devices: Minimum number of devices required
            min_houses: Minimum number of houses required
            activity_threshold: Minimum activity percentage

        Returns:
            dict: {"houses": [...], "devices": [...], "train_houses": [...], "test_houses": [...]}
        """
        availability = REDD_DataBuilder.get_device_availability(
            data_path, activity_threshold=activity_threshold
        )

        best_config = None
        best_score = 0

        for combo, devices in availability["common_devices"].items():
            if len(combo) >= min_houses and len(devices) >= min_devices:
                score = len(combo) * len(devices)
                if score > best_score:
                    best_score = score
                    best_config = {"houses": list(combo), "devices": devices}

        if best_config:
            houses = best_config["houses"]
            if len(houses) >= 3:
                best_config["train_houses"] = houses[:-1]
                best_config["test_houses"] = [houses[-1]]
            else:
                best_config["train_houses"] = houses[:1]
                best_config["test_houses"] = houses[1:]

        return best_config
