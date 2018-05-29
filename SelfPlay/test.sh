#!/usr/bin/env bash
PYTHONPATH=$PWD python3 -u app/mountaincar_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_max_steps_per_episode 100 --model_is_self_play False> log.txt
PYTHONPATH=$PWD python3 -u app/acrobot_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_max_steps_per_episode 100 --model_is_self_play False > log.txt
PYTHONPATH=$PWD python3 -u app/mazebase_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_use_baseline False --model_max_steps_per_episode 100 --model_is_self_play False > log.txt
PYTHONPATH=$PWD python3 -u app/cartpole_reinforce.py --model_num_epochs 6 --model_batch_size 2 --model_max_steps_per_episode 100 --model_use_baseline True --model_is_self_play False
PYTHONPATH=$PWD python3 -u app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40  --model_is_self_play True --model_is_self_play_with_memory True --model_memory_type lstm_memory > log.txt
PYTHONPATH=$PWD python3 -u app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40  --model_is_self_play True --model_is_self_play_with_memory True --model_memory_type base_memory > log.txt
PYTHONPATH=$PWD python3 -u app/selfplay.py --model_num_epochs 3 --model_batch_size 2 --model_max_steps_per_episode 40 --model_is_self_play True > log.txt
PYTHONPATH=$PWD python3 -u app/plot.py
