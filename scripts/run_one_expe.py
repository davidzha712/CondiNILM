"""Backward-compatibility stub. Use scripts.run_experiment instead."""
from scripts.run_experiment import *  # noqa: F401,F403

if __name__ == "__main__":
    from scripts.run_experiment import main
    main()
