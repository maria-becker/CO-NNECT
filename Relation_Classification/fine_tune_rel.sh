export DATA_DIR=./data/8bal-conf
export TRAIN_DATA=train-add-random-plus
export TEST_DATA=dev-add-random-plus
export MAX_LENGTH=15
export LEARNING_RATE=2e-5
export WEIGHT_DECAY=0.01
export BERT_MODEL=distilbert-base-uncased
export NUM_EPOCHS=3
export SEED=2
export OUTPUT_DIR_NAME=8bal-add-random-plus-3
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist
mkdir -p $OUTPUT_DIR
# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

python fine_tune_cnRel.py --data_dir $DATA_DIR \
--train_data $TRAIN_DATA \
--test_data $TEST_DATA \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--learning_rate $LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--num_train_epochs $NUM_EPOCHS \
--seed $SEED \
--do_train \
--evaluate_during_training \
--no_cuda