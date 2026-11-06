# Plan: evaluate the Spark with alphazero-general

This is the evaluation that follows [SPARK-HANDOVER.md](SPARK-HANDOVER.md). The goal is to see what this machine does with the games in this repo. It is not a search for a strong player, and it does not start SPaDE `di`.

Work in order. A game that fails a stage is recorded and left. The other games still go forward. Do not raise the workload for a game that has not passed the smaller stage.

Fill the result column at the end as each check finishes. One line is enough: pass, or the first error.

## How to run a check

Use the AlphaZero environment, not the SPaDE one. `python3` on this machine is `~/.local/spade-venv`.

```sh
export C_INCLUDE_PATH="$HOME/.local/python3.12-dev/include:$HOME/.local/python3.12-dev/include/python3.12"
export CFLAGS="-I$HOME/.local/python3.12-dev/include -I$HOME/.local/python3.12-dev/include/python3.12"
cd ~/git/alphazero-general
PY="$HOME/.local/az-venv/bin/python"
```

PyTorch is 2.14.1+cu130 in that environment. The wheel's compiled architectures stop at `sm_120`. This GPU is `sm_121`. The handover mentions a `torch<2.5` pin. This checkout's `requirements.txt` has no pin. Leave the installed wheel in place.

`train.py` is the driver for the comparable rounds. The files `alphazero/envs/*/train*.py` are a later stage. They have no command-line overrides, and their own numbers are much larger. Omitted flags on `train.py` keep the Coach defaults, including 1000 iterations and a game count of 256 times the CPU count. Pass the small budget in full every time.

Checkpoints, sample data, and TensorBoard logs land in `checkpoint/`, `data/`, and `runs/`. Those directories are gitignored. Prefix every run name with `spark-`. Keep console logs under `/tmp/spark-eval/`.

A training check passes when all of these are true:

- the workers start and the process exits on its own
- at least one self-play game finishes
- the training step runs
- a new file appears under `checkpoint/<run-name>/`
- the log has no CUDA re-initialisation error and no hang after the iteration banner

During one training step of stage 2, and again in stage 4, look at `nvidia-smi` and note whether this Python process is on the GPU.

## The resident model

A vLLM service is the heavy resident process. The tiny tic-tac-toe round already ran beside it.

Leave it up for stages 1 and 2. If one of those rounds dies for lack of memory, stop the service and repeat that round before calling the game a failure.

Stop it before stage 4, and leave it down through stage 6:

```sh
cd ~/.local/share/llm-stack
docker compose --profile serve down
```

Bring it back only after stage 6, or sooner if some other task needs the model:

```sh
cd ~/.local/share/llm-stack
docker compose --profile serve up -d
```

## Small budget

Use this for every game in stage 2. Two workers, one iteration, four games, ten simulations, a short training step, and a four-game baseline match. Self-play uses the net from the first iteration, so the GPU path is exercised. Past-model comparison and model gating stay off until stage 3.

```sh
mkdir -p /tmp/spark-eval
"$PY" train.py --env ENV \
  --run-name spark-ENV \
  --workers 2 \
  --numIters 1 \
  --numWarmupIters 0 \
  --gamesPerIteration 4 \
  --numMCTSSims 10 \
  --process-batch-size 8 \
  --train-batch-size 32 \
  --train-steps-per-iteration 4 \
  --no-autoTrainSteps \
  --arenaCompare 4 \
  --arenaCompareBaseline 4 \
  --no-arenaMCTS \
  --no-compareWithPast \
  --no-model-gating \
  --no-train-on-past-data \
  2>&1 | tee /tmp/spark-eval/ENV-tiny.log
```

Replace `ENV` with the environment name. For the old tafl engine the environment is still `hnefatafl`, and the module override is added:

```sh
--game-module alphazero.envs.hnefatafl.tafl_old --run-name spark-tafl-old
```

## Stage 0. Already done

On 5 November 2026 the GB10 was visible to PyTorch 2.14.1+cu130, Connect4 imported, and one tiny tic-tac-toe round trained on the GPU and wrote checkpoints. Those checkpoints were under `/tmp/az-smoke` and are not kept. Do not reinstall PyTorch for this evaluation. Repeat tic-tac-toe in stage 2 so the checkpoints from this plan sit with the others.

## Stage 1. Compile, and play without training

Import each engine and play random legal moves until the game ends or 40 moves have passed. This is the compile check. No network.

Order:

1. tictactoe
2. connect4
3. brandubh, which is the 7×7 game in `alphazero/envs/brandubh/fastafl.pyx`
4. othello, 8×8
5. hnefatafl, which is the 11×11 game in `alphazero/envs/hnefatafl/fastafl.pyx` and the module `train.py --env hnefatafl` loads
6. gobang, 15×15
7. `tafl_old`, loaded only by `alphazero/envs/hnefatafl/train.py`
8. stratego

`alphazero/envs/hnefatafl/brandubh.pyx` is not what the brandubh train script loads. Import it once and record whether it compiles. It does not need a training round.

These are expected to fail. Record the traceback and move on:

- `chess`. `alphazero/envs/chess/chess.py` is a stub and does not define `Game`. `alphazero/envs/chess/train.py` is empty.
- `alphazero/envs/hnefatafl/train_test.py`, which imports `alphazero.envs.hnefatafl.tafl`. That module is not in the tree. Do not use this script as the tiny run.
- `alphazero/envs/connect4/test_connect4.py` and `alphazero/envs/connect4/pit.py`, which import `Connect4Game`. The current module is `connect4.pyx`.
- The GUI under `AlphaZeroGUI` and the per-game `gui.py` files. PySide is not installed.

The old top-level trees `boardgame/`, `fastafl/`, and `hnefatafl/` are not on the trainer path. Leave them out.

## Stage 2. One tiny round of each runnable game

Run the small budget in the stage 1 order, skipping anything that did not import.

Tic-tac-toe has already passed once, on 5 November 2026, with checkpoints that were left under `/tmp/az-smoke` and are not kept. Repeat it as `spark-tictactoe` so this evaluation has its own checkpoint.

Gobang's stock script sets `train_on_past_data`. The small-budget command turns that off. Keep it off. There is no old gobang data in this clone.

## Stage 3. A second iteration, a resume, and a pit

Use connect4, still at the small game count.

1. Two iterations, with `--compareWithPast` and `--numIters 2`. Iteration 2 must train on the saved samples and play a short match against the previous net.
2. Resume. Run one iteration under `--run-name spark-connect4-resume`, stop the process after `iteration-0001` is written, then start the same command again with `--numIters 2`. The second start must continue from the saved net and finish iteration 2. `--load-model` is the default.
3. One iteration with `--model-gating` and `--arenaMCTS`, still four games. This is only to show those branches run.
4. `pit.py` for four games of `RawMCTSPlayer` against `random` on connect4, then four games of a network player loaded from the stage 2 connect4 checkpoint against `random`.

## Stage 4. Workers, with the resident model stopped

Stop vLLM first.

The net runs in the parent process. The workers are forked search processes. The question from the handover is whether that arrangement survives on this GPU.

1. Repeat the connect4 small budget with `--workers 2`. Confirm there is no CUDA fork error.
2. The same, with `--workers 8` and `--gamesPerIteration 8`.
3. Sample `nvidia-smi` once while self-play is asking the net for a batch, and once during the training step.

If the workers die or the run hangs at the iteration banner, stop. Record that, and do not start stage 5. Changing the process start method would be a separate fix.

## Stage 5. One scaled round, then the stock scripts at a cap

Scaled round, othello only, after stage 4 has passed:

- `--workers 8`
- `--numIters 3`
- `--gamesPerIteration 64`
- `--numMCTSSims 50`
- `--process-batch-size 32`
- `--arenaCompare 16` and `--arenaCompareBaseline 16`
- `--arenaMCTS`
- `--compareWithPast` from iteration 2

If othello fails, use connect4 at this same budget. Do not scale a second game in this stage.

Then run each stock script through a wrapper. Do not edit the scripts. The wrapper imports the script's arguments, forces the caps below, and calls `learn()`. The point is the module and the network shape that script selects, not its published game count.

Caps for every stock script:

- one iteration
- four workers
- eight games
- 25 simulations where the script asks for more
- four arena games
- `train_on_past_data` off
- `skipSelfPlayIters` cleared

Scripts, in this order:

| Script | What it actually loads |
| --- | --- |
| `alphazero/envs/tictactoe/train.py` | Tic-tac-toe. The file sets `skipSelfPlayIters=1`, which skips self-play on a first run. The wrapper clears it. |
| `alphazero/envs/connect4/train.py` | Connect4. The file asks for every CPU and 2048 games times that count, for 1000 iterations. |
| `alphazero/envs/othello/train.py` | Othello, 512 games and 100 simulations in the file. |
| `alphazero/envs/gobang/train.py` | Gobang. The file sets `train_on_past_data`. The wrapper turns it off. The network in the file is wider and deeper than the `train.py` default. |
| `alphazero/envs/hnefatafl/train_brandubh.py` | Brandubh via `alphazero.envs.brandubh.fastafl`. |
| `alphazero/envs/hnefatafl/train.py` | `tafl_old`, not the fastafl hnefatafl module. |
| `alphazero/envs/hnefatafl/train_fastafl.py` | 11×11 fastafl. The network in the file is the largest of these scripts. |
| `alphazero/envs/stratego/train.py` | Stratego. |

`train_test.py` stays in the expected-failure list from stage 1.

## Stage 6. Leave one run up for hours

Start this only after the scaled othello round, or its connect4 substitute, has finished cleanly. vLLM stays down. One run is enough.

Use `train.py --env othello` with the stage 5 worker and simulation counts, `--gamesPerIteration 128`, and a high `--numIters`. Stop at eight hours, or sooner if an iteration fails, a CUDA error appears, or thirty minutes pass with no finished game.

Record how many iterations finished, whether the loss stayed a finite number, and whether the process was still running at the stop. This is the handover's question about staying up for hours. Do not add a second long game.

## Out of this evaluation

Playing strength is not a result. A fresh net is expected to lose.

Chess, the GUI, `agprop`, and SPaDE `di` stay out. A Spark-specific break that stops every later stage is recorded and the evaluation stops there. Patching it is a separate decision.

## Results

| Check | Result |
| --- | --- |
| Stage 0. GPU visible, PyTorch 2.14.1+cu130, GB10 | Pass, 5 Nov 2026. Architectures in the wheel stop at sm_120. |
| Stage 0. Connect4 import | Pass, 5 Nov 2026. |
| Stage 1. tictactoe random play | Pass. 8 moves, game ended. |
| Stage 1. connect4 random play | Pass. 9 moves, game ended. |
| Stage 1. brandubh random play | Pass. 40 moves, still going. |
| Stage 1. othello random play | Pass. 40 moves, still going. |
| Stage 1. hnefatafl random play | Pass. 40 moves, still going. |
| Stage 1. gobang random play | Pass. 40 moves, still going. |
| Stage 1. tafl_old random play | Fail. Compiled, then `ImportError: cannot import name Board` from `hnefatafl.engine`. |
| Stage 1. stratego random play | Fail. `NotImplementedError: Stratego env is not currently functional`. |
| Stage 1. hnefatafl/brandubh.pyx import | Fail. Compiled, then the same missing `Board` import from `hnefatafl.engine`. |
| Stage 1. chess stub | Expected failure. `ModuleNotFoundError: No module named 'chess'`. |
| Stage 1. train_test.py missing `tafl` | Expected failure. `No module named 'alphazero.envs.hnefatafl.tafl'`. |
| Stage 1. Connect4Game imports | Expected failure. `test_connect4.py` and running `connect4/pit.py` both stop on `No module named 'alphazero.envs.connect4.Connect4Game'`. PySide6 is not installed, so the GUI stays out. |
| Stage 2. spark-tictactoe | Pass. Checkpoint `iteration-0001.pkl`. The training process was on the GPU, about 300 MiB, beside the resident model. No CUDA re-initialisation error. |
| Stage 2. spark-connect4 | Pass. Checkpoint written. |
| Stage 2. spark-brandubh | Pass. Checkpoint written. |
| Stage 2. spark-othello | Pass. Checkpoint written. |
| Stage 2. spark-hnefatafl | Fail. `Game` has no `has_draw`, so the network is not built. |
| Stage 2. spark-gobang | Pass. Past-data training stayed off. Checkpoint written. |
| Stage 2. spark-tafl-old | Skipped. Stage 1 did not import. |
| Stage 2. spark-stratego | Skipped. Stage 1 did not construct a game. |
| Stage 3. connect4 two iterations with past comparison | Pass. `spark-connect4-past` finished iterations 1 and 2, wrote `iteration-0002.pkl`, and both iterations played four games against iteration 0. |
| Stage 3. connect4 resume | Pass. `spark-connect4-resume` wrote iterations 0 and 1, then a second start continued at iteration 2 and wrote `iteration-0002.pkl`. |
| Stage 3. connect4 gating and arena MCTS | Pass. `spark-connect4-gate` played the baseline and past matches with arena search, then printed `Using model version 1 for self play.` |
| Stage 3. pit.py random and checkpoint | Raw search passed under the name `raw_mcts`: four games, 2 wins each. The plan's name `RawMCTSPlayer` is rejected. The stage 2 network loads, then its moves are `numpy.int64`, which this Arena loop discards, so those four games are not scored. |
| Stage 4. vLLM stopped | |
| Stage 4. connect4, 2 workers, no CUDA fork error | |
| Stage 4. connect4, 8 workers | |
| Stage 4. nvidia-smi during self-play and training | |
| Stage 5. scaled othello, 3 iterations | |
| Stage 5. stock tictactoe, capped | |
| Stage 5. stock connect4, capped | |
| Stage 5. stock othello, capped | |
| Stage 5. stock gobang, capped, past data off | |
| Stage 5. stock brandubh, capped | |
| Stage 5. stock tafl_old, capped | |
| Stage 5. stock fastafl, capped | |
| Stage 5. stock stratego, capped | |
| Stage 6. multi-hour othello | |
| Stage 6. vLLM restored, if it was still wanted | |
