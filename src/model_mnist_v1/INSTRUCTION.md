# Instruction

## Some links
- [link](https://medium.com/google-cloud/hello-world-on-gcp-ml-engine-cc09f506361c)

## Definition
    task.py contains the trainer logic that manages the job.

    model.py contains the TensorFlow graph code—the logic of the model.

    util.py if present, contains code to run the trainer.
    
## Instruction
### train_local
MODEL_DIR=./output
TRAIN_DATA=./train
EVAL_DATA=./eval

 gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir ${MODEL_DIR} \
    -- \
    --train-files ${TRAIN_DATA} \
    --eval-files ${EVAL_DATA} \
    --train-steps 1000 \
    --eval-steps 100

### 