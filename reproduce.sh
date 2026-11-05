#!/usr/bin/env bash
# Training runs behind results/final_returns.csv, grouped by task. Logs go to runs/.
# Usage: bash reproduce.sh hopper|walker2d|halfcheetah|ant|frozen|humanoid
# Then:  python analysis/extract.py --log_dir runs --out runs/summary
#        python analysis/table.py --csv runs/summary/final_returns.csv
# Each run is one process that trains all its agents in one batch on the CPU (about 50 batched steps per second on
# machine A, so 2M steps take about 11 hours). Runs are independent and can be started in parallel.
# The tags match the log names in analysis/extract.py. Keep --rules, --seed0 and --n_seeds as they are: an agent's
# action noise depends on --seed0 and on its position in the batch. --legacy_init marks the runs made with the
# earlier initialisation (README, Bitwise reproduction).
set -euo pipefail
train() { python -m slc.train --out runs --resume "$@"; }

case "${1:-}" in
  hopper)
    train --env Hopper-v5 --rules bp,bpsc,fsign --seed0 0 --n_seeds 5 --steps 2000000 --legacy_init --tag r1
    train --env Hopper-v5 --rules bp,bpsc,fsign --seed0 5 --n_seeds 5 --steps 2000000 --legacy_init --tag r1s5
    train --env Hopper-v5 --rules dfa,sign,kp --seed0 5 --n_seeds 5 --steps 2000000 --kp_decay 0 --tag f1a
    train --env Hopper-v5 --rules dfa,sign --seed0 0 --n_seeds 5 --steps 2000000 --tag f1b
    ;;
  walker2d)
    train --env Walker2d-v5 --rules bp,bpsc,fsign --seed0 0 --n_seeds 5 --steps 3000000 --legacy_init --tag r1
    train --env Walker2d-v5 --rules bp,bpsc,fsign --seed0 5 --n_seeds 5 --steps 3000000 --legacy_init --tag r1s5
    train --env Walker2d-v5 --rules dfa,sign,kp --seed0 5 --n_seeds 5 --steps 3000000 --kp_decay 0 --tag f2a
    train --env Walker2d-v5 --rules dfa,sign --seed0 0 --n_seeds 5 --steps 3000000 --tag f2b
    # kp seeds 0-4; this run also trained fkp (not reported)
    train --env Walker2d-v5 --rules fkp,kp --seed0 0 --n_seeds 5 --steps 3000000 --kp_decay 0 --tag p2d0
    ;;
  halfcheetah)
    train --env HalfCheetah-v5 --rules bp,bpsc,fsign --seed0 0 --n_seeds 5 --steps 2000000 --legacy_init --tag r1
    train --env HalfCheetah-v5 --rules bp,bpsc,fsign --seed0 5 --n_seeds 5 --steps 2000000 --tag r1s5
    train --env HalfCheetah-v5 --rules dfa,sign,kp --seed0 0 --n_seeds 5 --steps 2000000 --kp_decay 0 --tag f3a
    train --env HalfCheetah-v5 --rules dfa,sign,kp --seed0 5 --n_seeds 5 --steps 2000000 --kp_decay 0 --tag f3b
    ;;
  ant)
    train --env Ant-v5 --rules bp,bpsc,fsign --seed0 0 --n_seeds 3 --steps 2000000 --legacy_init --tag r1
    # seeds 3-5; this run also trained fkp and kp with weight decay 1e-5 (not reported)
    train --env Ant-v5 --rules bp,bpsc,fsign,fkp,kp --seed0 3 --n_seeds 3 --steps 2000000 --kp_decay 1e-5 --tag s3
    train --env Ant-v5 --rules bp,bpsc,fsign --seed0 6 --n_seeds 4 --steps 2000000 --tag s6
    train --env Ant-v5 --rules dfa,sign,kp --seed0 0 --n_seeds 5 --steps 2000000 --kp_decay 0 --tag f4a
    train --env Ant-v5 --rules dfa,sign,kp --seed0 5 --n_seeds 5 --steps 2000000 --kp_decay 0 --tag f4b
    ;;
  frozen)
    # 3-seed control with frozen hidden layers. The reported run came from an earlier single-file version of the
    # trainer that also trained fa (feedback alignment, not included here) and drew the action noise differently;
    # these commands repeat the experiment, not the exact random streams.
    for env in Hopper-v5 Walker2d-v5 HalfCheetah-v5 Ant-v5; do
      train --env "$env" --rules bp,dfa,sign,frozen --seed0 0 --n_seeds 3 --steps 2000000 --tag p1
    done
    ;;
  humanoid)
    # sign-agreement check; then: python analysis/arcsine_check.py runs/Humanoid-v5_humT5.json
    train --env Humanoid-v5 --rules fsign,sign,dfa --seed0 0 --n_seeds 3 --steps 500000 --shadow --log_every 10000 --tag humT5
    ;;
  *)
    echo "usage: bash reproduce.sh hopper|walker2d|halfcheetah|ant|frozen|humanoid" >&2
    exit 1
    ;;
esac
