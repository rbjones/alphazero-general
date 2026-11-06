# Handover: AlphaZero-general on DGX Spark

Continue from the Mac Grok session (workspace `/Users/rbj/git`, session about alphazero-general + SPaDE di + Spark).

## What the user wants (plain language)

1. **Not building SPaDE `di` yet.** Deductive intelligence waits until there are prototypes of `kr` and `dk`.
2. **Use the Spark a bit now** by running the existing AlphaZero-general games, to learn the machine and see what breaks.
3. **Keep explanations simple.** Avoid hyperparameter dumps unless asked.
4. **Later (not now):** adapt an existing AlphaZero-style theorem prover for HOL. Even later: propositional `agprop` as a first logic env.

## What this repo is

`alphazero-general` trains a neural net by playing a game against itself. Out of the box the net is random. Strength comes only after training. Weights are **not** in git.

This Mac had Python 3.12, **no PyTorch, no Cython**. Training has to happen on the Spark.

## What to run (the “thrash”)

Working train scripts (from repo root):

- `python3 -m alphazero.envs.tictactoe.train`
- `python3 -m alphazero.envs.connect4.train`
- `python3 -m alphazero.envs.othello.train`
- `python3 -m alphazero.envs.gobang.train`
- `python3 -m alphazero.envs.stratego.train`
- hnefatafl variants: `train.py`, `train_brandubh.py`, `train_fastafl.py`, `train_test.py`

**Do not treat as runnable yet**

- `chess/` — unfinished stub
- `agprop/` — template only, not wired to AlphaZero

**Known traps**

- Install a **current** CUDA PyTorch for the Spark (aarch64). Ignore `requirements.txt` pin `torch<2.5`.
- First import compiles Cython (`.pyx` files).
- Tic-tac-toe train has `skipSelfPlayIters=1` (expects old data). Clear that on a first run.
- Gobang has `train_on_past_data=True` (same issue).
- Connect4 defaults are very large (many games per round). Start smaller, then scale up.
- Skip the GUI for now (`AlphaZeroGUI` / PySide2).

## Suggested first work on Spark

The first checks are done: the GPU is visible, Connect4 imports, and one tiny tic-tac-toe round wrote a checkpoint. Continue with [SPARK-EVAL-PLAN.md](SPARK-EVAL-PLAN.md).

Record, for each later stage: did it compile, did CUDA work while workers were running, did a round finish and write a checkpoint, did a run stay up for hours.

## SPaDE context (do not implement now)

`SPaDE/di` is the future home of “focal” AlphaZero-like search over logical contexts. `di/agprop` is an empty propositional stub. User was clear: **di cannot start before kr and dk prototypes.**
