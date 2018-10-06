#!/bin/bash

DATA_PATH='../../../../simple-examples/data/'
MODEL='custom'

EPOCH=25
HIDDEN_SIZE_LIST=(50)
LR_LIST=(1.0)
LR_DECAY_LIST=(0.5)
BATCH_SIZE_LIST=(20)
EXP_LIST=(1.0)
LOSS_FUNC='logistic'
NUM_STEPS=20

for LR in "${LR_LIST[@]}"
do
  for BATCH_SIZE in "${BATCH_SIZE_LIST[@]}"
  do
    for HIDDEN_SIZE in "${HIDDEN_SIZE_LIST[@]}"
    do
      for LR_DECAY in "${LR_DECAY_LIST[@]}"
      do
        for EXP in "${EXP_LIST[@]}"
        do
          SAVE_PATH='saved_models/ptb_LR'$LR'_BS'$BATCH_SIZE'_HS'$HIDDEN_SIZE'_LD'$LR_DECAY'_EXP'$EXP
          echo 'python ptb_word_lm.py --data_path='$DATA_PATH' --model='$MODEL' --learning_rate='$LR' --lr_decay='$LR_DECAY' --exp='$EXP' --batch_size='$BATCH_SIZE' --max_max_epoch='$EPOCH' --hidden_size='$HIDDEN_SIZE' --loss_func='$LOSS_FUNC' --num_steps='$NUM_STEPS' --save_path='$SAVE_PATH
          python ptb_word_lm.py --data_path=$DATA_PATH --model=$MODEL --learning_rate=$LR --lr_decay=$LR_DECAY --exp=$EXP --batch_size=$BATCH_SIZE --max_max_epoch=$EPOCH --hidden_size=$HIDDEN_SIZE --loss_func=$LOSS_FUNC --num_steps=$NUM_STEPS --save_path=$SAVE_PATH
          echo ''
          echo ''
          echo ''
          echo ''
          echo ''
          echo ''
	done
      done
    done
  done
done
