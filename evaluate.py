#!/usr/bin/env python

import os, json, argparse, pdb, numpy as np, pandas, time
from shapely.geometry import MultiPolygon, box
from TensorBox.predict import TensorBox
from subprocess import check_output
from precision_recall import precision_recall
from tqdm import tqdm

def get_results(args, model):
    all_predictions = []
    gt_boxes = json.load(open(args.test_boxes))

    for _, predictions, truth in tqdm(model.predict_all(args.test_boxes, args.threshold), total=len(gt_boxes)):
        all_predictions.append(predictions)

    df = pandas.DataFrame(pandas.concat(all_predictions))
    df.columns = ['x1', 'y1', 'x2', 'y2', 'score', 'image_id']

    recall, precision = precision_recall(gt_boxes, df)

    i = np.argmax(recall + precision)
    f1 = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    print('precision = %f, recall = %f, F1 = %f' % (precision[i], recall[i], f1))

    return df

def get_weights(model):
    download_url = 'https://github.com/ArnholdInstitute/ColdSpots/releases/download/1.0/%s.zip'
    if not os.path.exists('weights'):
        os.mkdir('weights')
    if not os.path.exists(os.path.join('weights', model)):
        print('Downloading weights for %s' % model)
        if not os.path.exists(os.path.join('weights', model + '.zip')):
            check_output(['wget', download_url % model, '-O', os.path.join('weights', model + '.zip')])
        print('Unzipping...')
        check_output(['unzip', 'weights/%s.zip' % model, '-d', 'weights'])
    return json.load(open('weights/%s/description.json' % model))

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--test_boxes', required=True, help='Path to the JSON file containing the test set')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--model', required=True, help='Which model to run (tensorbox, yolo, or rcnn)')
    parser.add_argument('--weights', default=None, help='Path to weight file (default is in description.json)')
    parser.add_argument('--threshold', default=None, type=float, help='Confidence threshold (default is in description.json)')
    args = parser.parse_args()

    description = get_weights(args.model)

    if args.weights is None:
        args.weights = os.path.join('weights/%s/%s' % (args.model, description['weights']))

    if args.threshold is None:
        args.threshold = description['threshold']

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    if not os.path.exists('results'):
        os.mkdir('results')

    model = None
    if args.model == 'tensorbox':
        model = TensorBox(args.weights)
    else:
        raise Exception('Model not recognized!')

    df = get_results(args, model)
    df.to_csv('results/%s_predictions.csv' % args.model, index=False)

if __name__ == '__main__':
    main()
