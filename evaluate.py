#!/usr/bin/env python

import os, json, argparse, pdb, numpy as np, pandas, time, importlib, psycopg2, boto3
from shapely.geometry import MultiPolygon, box
from subprocess import check_output
from precision_recall import precision_recall
from tqdm import tqdm
from model import MODELS
from zipfile import ZipFile
from db import aigh_conn
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

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

    summary = {
        'precision' : precision[i],
        'recall' : recall[i],
        'f1' : f1,
        'threshold' : float(df.sort_values(by='score', ascending=False)['score'].values[i]),
        'name' : args.model_class.name,
        'id' : args.model_class.mk_hash(args.weights),
        'weights' : os.path.basename(args.weights),
        'test_boxes' : args.test_boxes
    }

    return df, summary

def upload_model(args, summary):
    with aigh_conn.cursor() as cur:
        cur.execute("SELECT MAX(f1) FROM models WHERE name=%s", (summary['name'],))
        best_score, = cur.fetchone()
        if best_score and best_score > summary['f1']:
            print('Not uploading because, a model already exists with f1 %f' % best_score)
            return

    zip_file = args.model_class.zip_weights(args.weights, base_dir='weights')
    s3 = boto3.resource('s3')
    key = os.path.join('building-detection', os.path.basename(zip_file))

    qargs = {k : summary[k] for k in ['name', 'id', 'precision', 'recall', 'threshold', 'f1']}
    qargs['instance'] = '/'.join(args.weights.split('/')[-2:])
    qargs['s3_loc'] = os.path.join('s3://aigh-deep-learning-models/', key)

    conn = psycopg2.connect(
        dbname=os.environ.get('PGDATABASE', 'aigh'),
        user=os.environ.get('PGUSER', ''),
        password=os.environ.get('PGPASSWORD', ''),
        host=os.environ.get('PGHOST', '')
    )

    with aigh_conn.cursor() as cur:
        cur.execute("""
            INSERT INTO models(name, instance, id, tested_on, precision, recall, threshold, s3_loc, f1)
            VALUES (%(name)s, %(instance)s, %(id)s, now(), %(precision)s, %(recall)s, %(threshold)s, %(s3_loc)s, %(f1)s)
        """, qargs)

        json.dump(summary, open('.summary.json', 'w'))
        with ZipFile(zip_file, 'a') as z:
            z.write('.summary.json', '%s/description.json' % qargs['id'])

        s3.meta.client.upload_file(zip_file, 'aigh-deep-learning-models', key)
        aigh_conn.commit()

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--test_boxes', required=True, help='Path to the JSON file containing the test set')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--weights', default=None, help='Path to weight file (default is in description.json)')
    parser.add_argument('--threshold', default=0.5, type=float, help='Confidence threshold (default is in description.json)')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    if not os.path.exists('results'):
        os.mkdir('results')

    model = None
    for model_type in MODELS:
        if model_type in args.weights:
            module = importlib.import_module(model_type)
            args.model_class = getattr(module, module.NAME)
            model = args.model_class(args.weights)
            model_id = args.model_class.mk_hash(args.weights)

    if model is None:
        raise ValueError('Invalid model')

    df, summary = get_results(args, model)
    upload_model(args, summary)

if __name__ == '__main__':
    main()
