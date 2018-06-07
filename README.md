## PySC2 Reinforcement Learning Agents

Reinforcement Learning agents that play the StarCraft II Minimaps and
eventually will be able to play the full game.

This project depends on the [sc2gym OpenAI Gym Environments](https://github.com/islamelnabarawy/sc2gym).

This is still a work in progress. The `train_a2c.py` no longer runs due to changes in the `baselines` code. The current 
plan is to replace both `train_dqn.py` and `train_a2c.py` with standalone implementations, and add a standalone A3C
implementation, as well as benchmark results on the various `sc2gym` environments.
