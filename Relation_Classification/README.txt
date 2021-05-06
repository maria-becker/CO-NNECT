Establishing Commonsense Knowledge Paths between Concepts Via Relation Classification

This Readme explains the code for running COREC-LM for predicting direct relations between pairs of concepts, as described in our paper (Becker et al. 2021). COREC-LM is part of our framework CO-NNECT, which we propose for enriching texts with commonsense knowledge in the form of high-quality single- and multi-hop knowledge paths between concepts in texts. With the following code, you can generate single-hop commonsense knowledge paths between concepts. The code for generating multi-hop knowledge paths between concepts from texts can be found in the Forwardchaining directory of this repository.


## Data
### Example
    Relation labels: 
                    HasSubevent 0
                    Causes 1
                    UsedFor 2
                    CapableOf 3
                    HasProperty 4
                    HasPrerequisite 5
                    IsA 6
                    AtLocation 7

    source lines:   watch even news <mask> learn about current event
                    have food <mask> eat

    target lines:   1
                    5
    
    Save the source lines in train.src and the target lines in train.trg in the same directory.
    Same for validate and test data. 


## Model
[distilbert-base-uncased](https://arxiv.org/abs/1910.01108)

## Fine-tuning
    
    Make label file and define it in 
    LABEL_FILE = 'labels.txt'
    
    Change the following parameters in ./fine_tune_rel.sh
    DATA_DIR: the directory of the training data (where the train.src and train.trg files are)
    TRAIN_DATA: e.g. train 
    TEST_DATA: validate data, e.g. valid (if there are valid.src and valid.trg files in the directory)
    MAX_LENGTH: the maximum length of each line (chars)
    OUTPUT_DIR_NAME: the directory to save the fine-tuned model

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

## Prediction and Evaluation
    OUTPUT_DIR: is the directory of the fine-tuned model and where the predictions will be saved
    DATA_DIR: is the directory where the test data and the label file are

    Only get predictions (in case no labelled test data is available): 
    
        python fine_tune_cnRel.py 
        --data_dir $DATA_DIR \
        --train_data $TRAIN_DATA \
        --test_data $f \
        --model_name_or_path $BERT_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --do_max_prediction \
        --no_cuda
    
    Evaluation (in case labelled test data is available):
    --> Make sure there are both src and trg files for test_data
    
        python fine_tune_cnRel.py 
        --data_dir $DATA_DIR \
        --train_data $TRAIN_DATA \
        --test_data $f \
        --model_name_or_path $BERT_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --do_eval \
        --no_cuda

    Multiple label prediction and evaluation:
    Save the 'threshold.csv' in the data_dir.
    Alternatively give the threshold in fine_tune_cnRel.py (in the block of do_multiple_prediction)
    
        python fine_tune_cnRel.py 
        --data_dir $DATA_DIR \
        --train_data $TRAIN_DATA \
        --test_data $f \
        --model_name_or_path $BERT_MODEL \
        --output_dir $OUTPUT_DIR \
        --max_seq_length  $MAX_LENGTH \
        --do_multiple_prediction \
        --no_cuda

    do multiple predictions evalution:
        change 
        --do_multiple_prediction 
        to 
        --do_eval_multilabels


Our pretained model for CN-100k, including the Random class, can be downloaded from here:
https://drive.google.com/file/d/1wAcrWgfYUOdfTefTHZszYmtHKGQhqRee/view?usp=sharing 100k+rand

Our pretained model for CN-13 (including the 13 most frequent relations in CN-100k), including the Random class, can be downloaded from here:
https://drive.google.com/file/d/1A5PLNCKcCjxHNU_0uzzeuQgX7AVfuvTR/view?usp=sharing
13rel+rand  

    
