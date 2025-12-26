for app in WashingMachine Dishwasher Kettle Microwave Fridge; do
  uv run -m scripts.run_optuna_search \
    --dataset UKDALE \
    --appliance "$app" \
    --name_model NILMFormer \
    --sampling_rate 1min \
    --window_size 128 \
    --seed 0 \
    --n_trials 30
done