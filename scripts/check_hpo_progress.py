"""Check Optuna HPO study progress."""
import sys
import optuna

db_path = sys.argv[1] if len(sys.argv) > 1 else "optuna_studies/refit_hpo_v11_batch2048.db"
study_name = sys.argv[2] if len(sys.argv) > 2 else None

storage = f"sqlite:///{db_path}"

summaries = optuna.study.get_all_study_summaries(storage=storage)
for s in summaries:
    print(f"Study: {s.study_name}")
    print(f"  Trials: {s.n_trials}")
    print(f"  Best value: {s.best_trial.value if s.best_trial else 'N/A'}")
    if study_name is None:
        study_name = s.study_name

if study_name:
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.trials

    print(f"\n=== Study: {study_name} ===")
    print(f"Total trials: {len(trials)}")

    completed = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]
    running = [t for t in trials if t.state == optuna.trial.TrialState.RUNNING]
    pruned = [t for t in trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in trials if t.state == optuna.trial.TrialState.FAIL]

    print(f"  Completed: {len(completed)}")
    print(f"  Running: {len(running)}")
    print(f"  Pruned: {len(pruned)}")
    print(f"  Failed: {len(failed)}")

    if completed:
        best = study.best_trial
        print(f"\nBest trial #{best.number}: value={best.value:.4f}")
        print("  Params:")
        for k, v in sorted(best.params.items()):
            print(f"    {k}: {v}")

    print("\nAll completed trials:")
    for t in completed:
        print(f"  T{t.number}: value={t.value:.4f} (duration={t.duration})")
