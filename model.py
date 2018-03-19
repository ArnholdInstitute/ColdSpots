import os, boto3, re, pdb, botocore, zipfile, json, importlib
from db import aigh_conn as conn

MODELS = set([
    'TensorBox',
    'darkflow',
    'faster_rcnn_pytorch',
    'ssd_pytorch'
])

def get_best_model():
    '''
    Select the best model based on f-score
    '''
    with conn.cursor() as cur:
        cur.execute("""
            SELECT name, instance, id, s3_loc
            FROM models
            ORDER BY f1 DESC
            LIMIT 1
        """)
        name, instance, id, s3_loc = cur.fetchone()
        id = id.replace('-', '')
        if not os.path.exists(os.path.join('weights', id)):
            # Download weights
            print('Downloading weights for %s (%s)' % (name, id))
            s3 = boto3.resource('s3')
            res = re.search('s3://([^/]*)/(.*)$', s3_loc)
            bucket, key = res.group(1), res.group(2)
            s3.Bucket(bucket).download_file(key, 'weights/%s.zip' % id)
            print('Extracting weights...')
            with zipfile.ZipFile('weights/%s.zip' % id, 'r') as z:
                z.extractall('weights')
            os.remove('weights/%s.zip' % id)

        description = json.load(open('weights/%s/description.json' % id))
        module = importlib.import_module(description['name'])
        model_class = getattr(module, module.NAME)
        model = model_class(weights=os.path.join('weights', id, description['weights']))
        return model, id, description['threshold']