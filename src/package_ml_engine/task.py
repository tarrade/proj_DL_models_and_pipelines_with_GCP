"""
Parse arguments and call main function
"""
import os
import argparse
import shutil

from .utils import load_data
from .model_mnist import train_and_evaluate


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        help='GCS or local path to training data',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default='128'
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for',
        type=int,
        default='200'
    )
    parser.add_argument(
        '--hidden_units',
        help='List of hidden layer sizes to use for DNN feature columns',
        nargs='+',
        type=int,
        default=[128, 64, 32]
    )
    parser.add_argument(
        '--job_dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    # Eval arguments
    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default='10',
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Seconds between evaluations',
        default=300,
        type=int
    )

    args = parser.parse_args().__dict__

    OUTDIR = args['output_dir']
    ##########################################
    # Load Data in Memoery

  # #ToDo: replace numpy-arrays
    (x_train, y_train), (x_test, y_test) = load_data(
        rel_path=args['data_path'])
  
    x_train = parse_images(x_train)
    x_test = parse_images(x_test)

    y_train = parse_labels(y_train)
    y_test = parse_labels(y_test)

    # #######################################
    # # Train
    # ToDo execute outside from skript
    shutil.rmtree(OUTDIR, ignore_errors=True)  # start fresh each time
    train_and_evaluate(args)