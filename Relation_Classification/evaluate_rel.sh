export DATA_DIR=./debate_essays
export TRAIN_DATA=train-add-random
export TEST_DATA='pairs_dev_essays pairs_train_essays'
export MAX_LENGTH=15
export LEARNING_RATE=2e-5
export BERT_MODEL=distilbert-base-uncased
export SEED=2
export OUTPUT_DIR_NAME=100k-add-random
export CURRENT_DIR=${PWD}
export OUTPUT_DIR=${CURRENT_DIR}/${OUTPUT_DIR_NAME}

# Make output directory if it doesn't exist

# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

for f in $TEST_DATA
do
  python fine_tune_cnRel.py
  --data_dir $DATA_DIR \
  --train_data $TRAIN_DATA \
  --test_data $f \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --do_max_prediction \
  --no_cuda
done
