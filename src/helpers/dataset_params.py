"""Dataset-specific parameter management -- CondiNILM.

Author: Siyi Li
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class DatasetParamsManager:
    """
    Manages dataset-specific hyperparameters for NILM training.
    """

    def __init__(self, config_path: str = "configs/dataset_params.yaml"):
        """
        Initialize the parameter manager.

        Args:
            config_path: Path to the dataset parameters YAML file
        """
        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Dataset params file not found: {self.config_path}")
            self._config = {}
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        logger.info(f"Loaded dataset params from {self.config_path}")

    def get_dataset_config(self, dataset: str) -> Dict[str, Any]:
        """
        Get full configuration for a dataset.

        Args:
            dataset: Dataset name (e.g., "UKDALE", "REDD", "REFIT")

        Returns:
            Dataset configuration dictionary
        """
        # Case-insensitive lookup
        dataset_upper = dataset.upper()
        for key in self._config:
            if key.upper() == dataset_upper:
                return self._config[key]
        return {}

    def get_available_appliances(self, dataset: str) -> List[str]:
        """
        Get list of available appliances for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            List of appliance names
        """
        config = self.get_dataset_config(dataset)
        appliances = config.get("appliances", {})
        return list(appliances.keys())

    def get_appliance_params(
        self, dataset: str, appliance: str
    ) -> Dict[str, Any]:
        """
        Get parameters for a specific appliance in a dataset.

        Args:
            dataset: Dataset name
            appliance: Appliance name

        Returns:
            Appliance parameters dictionary
        """
        config = self.get_dataset_config(dataset)
        appliances = config.get("appliances", {})

        # Case-insensitive lookup
        appliance_lower = appliance.lower()
        for key, params in appliances.items():
            if key.lower() == appliance_lower:
                return params
        return {}

    def get_appliance_houses(self, dataset: str, appliance: str) -> List[int]:
        """
        Get list of houses that have a specific appliance.

        Args:
            dataset: Dataset name
            appliance: Appliance name

        Returns:
            List of house indices
        """
        params = self.get_appliance_params(dataset, appliance)
        return params.get("houses", [])

    def get_training_config(self, dataset: str) -> Dict[str, Any]:
        """
        Get training configuration for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Training configuration dictionary
        """
        config = self.get_dataset_config(dataset)
        return config.get("training", {})

    def get_loss_config(self, dataset: str) -> Dict[str, Any]:
        """
        Get loss function configuration for a dataset.

        Args:
            dataset: Dataset name

        Returns:
            Loss configuration dictionary
        """
        config = self.get_dataset_config(dataset)
        return config.get("loss", {})

    def get_postprocess_config(
        self, dataset: str, appliance: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get postprocess configuration for a dataset or specific appliance.

        Args:
            dataset: Dataset name
            appliance: Optional appliance name

        Returns:
            Postprocess configuration dictionary
        """
        config = self.get_dataset_config(dataset)
        postprocess = config.get("postprocess", {})

        if appliance is None:
            return postprocess

        # Case-insensitive lookup
        appliance_lower = appliance.lower()
        for key, params in postprocess.items():
            if key.lower() == appliance_lower:
                return params
        return {}

    def get_common_config(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get common/shared configuration.

        Args:
            key: Optional specific key to retrieve

        Returns:
            Common configuration dictionary
        """
        common = self._config.get("common", {})
        if key is not None:
            return common.get(key, {})
        return common

    def get_device_type_params(self, device_type: str) -> Dict[str, Any]:
        """
        Get loss parameters for a specific device type.

        Args:
            device_type: Device type string

        Returns:
            Device type parameters dictionary
        """
        common = self.get_common_config("device_type_params")
        return common.get(device_type, {})

    def apply_dataset_config(
        self, expes_config: Any, dataset: str, appliances: Optional[List[str]] = None
    ) -> Any:
        """
        Apply dataset-specific configuration to experiment config.

        Args:
            expes_config: Experiment configuration object (OmegaConf or dict)
            dataset: Dataset name
            appliances: Optional list of appliances (auto-detect if None)

        Returns:
            Updated experiment configuration
        """
        ds_config = self.get_dataset_config(dataset)
        if not ds_config:
            logger.warning(f"No config found for dataset: {dataset}")
            return expes_config

        # Apply training config
        training = ds_config.get("training", {})
        for key, value in training.items():
            if key == "learning_rate":
                # Special handling for nested model_training_param
                pass  # Handled separately
            elif not hasattr(expes_config, key) or getattr(expes_config, key) is None:
                try:
                    expes_config[key] = value
                except (TypeError, KeyError):
                    setattr(expes_config, key, value)

        # Apply loss config
        loss_config = ds_config.get("loss", {})
        for key, value in loss_config.items():
            try:
                expes_config[key] = value
            except (TypeError, KeyError):
                setattr(expes_config, key, value)

        # Build appliance_param from dataset config
        appliance_param = {}
        ds_appliances = ds_config.get("appliances", {})
        for app_name, app_config in ds_appliances.items():
            appliance_param[app_name.lower()] = {
                "min_threshold": app_config.get("min_threshold", 50),
                "max_threshold": app_config.get("max_threshold", 3000),
                "min_on_duration": app_config.get("min_on_duration", 1),
                "min_off_duration": app_config.get("min_off_duration", 0),
                "min_activation_time": app_config.get("min_activation_time", 1),
            }

        # Store for later use
        try:
            expes_config["dataset_appliance_param"] = appliance_param
        except (TypeError, KeyError):
            pass

        logger.info(f"Applied {dataset} config: {len(training)} training params, {len(loss_config)} loss params")
        return expes_config


def auto_detect_appliances_from_data(data_path: str, dataset: str) -> List[str]:
    """
    Auto-detect available appliances from data files.

    Args:
        data_path: Path to dataset directory
        dataset: Dataset name

    Returns:
        List of detected appliance names
    """
    import glob

    detected = []

    if dataset.upper() == "REDD":
        # Check REDD CSV files for available columns
        pattern = os.path.join(data_path, "REDD", "redd_house*_0.csv")
        csv_files = glob.glob(pattern)

        all_columns = set()
        for csv_file in csv_files[:3]:  # Check first 3 houses
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, nrows=0)
                all_columns.update(df.columns.tolist())
            except Exception:
                continue

        # Map REDD column names to standard names
        column_mapping = {
            "dish washer": "dishwasher",
            "washer dryer": "washing_machine",
            "fridge": "fridge",
            "microwave": "microwave",
            "electric stove": "cooker",
            "electric space heater": "electric_heater",
            "electric furnace": "electric_furnace",
        }

        for col, std_name in column_mapping.items():
            if col in all_columns:
                detected.append(std_name)

    elif dataset.upper() == "UKDALE":
        # Check UKDALE labels files
        for house_id in range(1, 6):
            labels_path = os.path.join(data_path, "UKDALE", f"house_{house_id}", "labels.dat")
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                appliance = parts[1]
                                if appliance not in detected and appliance != "aggregate":
                                    detected.append(appliance)
                except Exception:
                    continue

    return detected


def get_dynamic_output_channels(
    appliances: Union[str, List[str]],
    dataset: str,
    params_manager: Optional[DatasetParamsManager] = None,
) -> int:
    """
    Get the number of output channels based on appliance selection.

    Args:
        appliances: Single appliance name, list of appliances, or "multi"/"all"
        dataset: Dataset name
        params_manager: Optional parameter manager instance

    Returns:
        Number of output channels (devices)
    """
    if params_manager is None:
        params_manager = DatasetParamsManager()

    if isinstance(appliances, str):
        if appliances.lower() in ("multi", "all"):
            # Use all available appliances
            available = params_manager.get_available_appliances(dataset)
            return len(available)
        else:
            # Single appliance
            return 1

    # List of appliances
    return len(appliances)


def validate_appliances_for_dataset(
    appliances: List[str],
    dataset: str,
    params_manager: Optional[DatasetParamsManager] = None,
) -> List[str]:
    """
    Validate and filter appliances that are available in the dataset.

    Args:
        appliances: List of requested appliances
        dataset: Dataset name
        params_manager: Optional parameter manager instance

    Returns:
        List of valid appliances
    """
    if params_manager is None:
        params_manager = DatasetParamsManager()

    available = params_manager.get_available_appliances(dataset)
    available_lower = {a.lower(): a for a in available}

    valid = []
    for app in appliances:
        app_lower = app.lower()
        if app_lower in available_lower:
            valid.append(available_lower[app_lower])
        else:
            logger.warning(f"Appliance '{app}' not available in {dataset}, skipping")

    return valid


# Global instance for convenience
_default_manager: Optional[DatasetParamsManager] = None


def get_default_manager() -> DatasetParamsManager:
    """Get the default parameter manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = DatasetParamsManager()
    return _default_manager
