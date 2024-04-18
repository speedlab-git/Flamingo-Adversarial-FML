#!/bin/bash
python main.py --dataset_root "/path/to/dataset" \
--data_split "non_iid" \
--optimizer "adam" \
--num_rounds 400 \
--clients_per_round 10 \
--batch_size 4 \
--num_epochs 50 \
--learning_rate 0.003 \
--ways 5 \
--shots 5 \
--meta_lr 0.01 \
--lambda 1\
