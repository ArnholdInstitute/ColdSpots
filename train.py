import argparse
import os
from TensorBox import train
import json

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default='0', help='Determines whether or not to run on GPU')
  parser.add_argument('--root_dir', default='../ColdSpots/data', help='Root directory for training data')
  parser.add_argument('--training_data', default='', help='Location of training data')
  args = parser.parse_args()
      
  # Load the default training arguments and hyperparameters.
  with open('hypes.json') as json_file:
    hyperparams = json.load(json_file)

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

  if (args.training_data):
    hyperparams['data']['train_idl'] = os.path.join(args.root_dir, args.training_data)
  train.train(hyperparams, [])


