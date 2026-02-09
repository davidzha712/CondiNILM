#!/usr/bin bash

declare -a DATASETS_1=("REFIT")
declare -a APPLIANCES_1=("WashingMachine" "Dishwasher" "Kettle" "Microwave")

declare -a DATASETS_2=("UKDALE")
declare -a APPLIANCES_2=("WashingMachine" "Dishwasher" "Kettle" "Microwave" "Fridge")

declare -a MODELS_1=("BiLSTM" "FCN" "CNN1D" "UNetNILM" "DAResNet" "BERT4NILM" "DiffNILM" \
                     "TSILNet" "Energformer" "BiGRU" "STNILM" "NILMFormer")
declare -a WINDOW_SIZES_1=("128" "256" "512" "360" "720")

declare -a MODELS_2=("ConvNet" "ResNet" "Inception")
declare -a WINDOW_SIZES_2=("day" "week" "month")

# Run experiments
run_batch() {
  local -n arr_datasets=$1
  local -n arr_appliances=$2
  local -n arr_models=$3
  local -n arr_windows=$4

  for dataset in "${arr_datasets[@]}"; do
    for appliance in "${arr_appliances[@]}"; do
      for win in "${arr_windows[@]}"; do
        for model in "${arr_models[@]}"; do
          echo "Running: uv run -m scripts.run_experiment \
            --dataset $dataset \
            --sampling_rate 1min \
            --appliance $appliance \
            --window_size $win \
            --name_model $model"
          uv run -m scripts.run_experiment \
            --dataset "$dataset" \
            --sampling_rate "1min" \
            --appliance "$appliance" \
            --window_size "$win" \
            --name_model "$model"
        done
      done
    done
  done
}

#####################################
# Run all possible experiments
#####################################
run_batch DATASETS_1 APPLIANCES_1 MODELS_1 WINDOW_SIZES_1
run_batch DATASETS_1 APPLIANCES_1 MODELS_2 WINDOW_SIZES_2
run_batch DATASETS_2 APPLIANCES_2 MODELS_1 WINDOW_SIZES_1
run_batch DATASETS_2 APPLIANCES_2 MODELS_2 WINDOW_SIZES_2
