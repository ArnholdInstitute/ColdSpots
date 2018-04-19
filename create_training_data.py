#!/usr/bin/env python

import os, json, argparse, pdb, numpy as np, psycopg2, cv2, pandas, boto3, math, itertools
from tqdm import tqdm
from shapely.geometry import MultiPolygon, box, mapping, shape
from functools import partial
from augmentations import rotate, noop, mirror, distort, crop
from ensemble_agreement import agree
from shapely import wkb
from skimage import io
from model import get_best_model
from db import aigh_conn as conn
from datetime import datetime

s3 = boto3.client('s3')

def get_gsd(lat, zoom):
    """
    Computes the Ground Sample Distance (GSD).  More details can be found
    here: https://msdn.microsoft.com/en-us/library/bb259689.aspx

    Args:
        lat : float - latitude of the GSD of interest
        zoom : int - zoom level (WMTS)
    """
    return (math.cos(lat * math.pi / 180) * 2 * math.pi * 6378137) / (256 * 2**zoom)

def get_image(file, data_dir = '../data'):
    '''
    Fetch an image from the S3 bucket if it doesn't already exist on the file system
    Arguments:
        file (str): The filename of the image (training_data/<country>/<filename>.jpg)
        data_dir (str): directory where images live/should be saved to
    '''
    path = os.path.join(data_dir, file)
    if os.path.exists(path):
        return cv2.imread(path)
    else:
        # Image doesn't exist on file system, fetch from S3...
        params = {'Bucket' : 'dg-images', 'Key' : file}
        url = s3.generate_presigned_url(ClientMethod='get_object', Params=params)
        # Convert from RGB -> BGR
        img = io.imread(url)[:,:,(2,1,0)]
        cv2.imwrite(path, img) # cache the image
        return img

def process_file(file, cur, augs, model, threshold, ensemble):
    '''
    Infer buildings for each augmentation of an image, transform 
    each prediction back to the original image space, and then run
    ensemble agreement.

    Arguments:
        ensemble (boolean): Whether to run ensemble agreement or not
        file (str): filename of the image to infer
        cur (psycopg2.cursor): database cursor
        augs (list): list of augmentation functions
        model: The model used to infer buildings
        threshold (float): Confidence threshold for a prediction to be considered positive

    Returns:
        An optional tuple containing three elements:
            - features (pandas.DataFrame): Columns - x1, y1, x2, y2  (box coordinates)
            - i (int): row offset for the 500 x 500 pixel subset of the image
            - j (int): column offset for the 500 x 500 pixel subset of the image
            - sample (np.ndarray): The subset of the image that was inferred
    '''
    img = get_image(file)[:-25, :, :] # strip BING logo
    # Iterate over the image in 500 x 500 pixel subsets (size the model accepts)
    for i, j in tqdm(list(itertools.product(range(0, img.shape[0], 500), range(0, img.shape[1], 500)))):
        sample = img[i:i+500, j:j+500, :]

        boxes = []

        # Iterate over each augmentation of the sample
        for k, augmentation in enumerate(augs):
            # returns a function to project and unproject a sample
            augmented, unaugment = augmentation(sample)

            current = model.predict_image(augmented)
            current = current[current.score >= threshold]
            current['augment_id'] = k
            transformed_boxes = unaugment(current)
            boxes.append(transformed_boxes)

        df = pandas.concat(boxes)
        # If we have positive predictions for more than one augmentation, run consensus
        if ensemble and df.augment_id.unique().shape[0] > 1:
            features = agree(df)
            # If we have agreement on predictions, return a result
            if len(features['features']) > 0:
                return features, i, j, sample
        elif not ensemble and len(df) > 0:
            features = []
            for _, r in df.round().astype(int).iterrows():
                features.append({'geometry' : mapping(box(r.x1, r.y1, r.x2, r.y2)), 'properties' : {}, 'type' : 'Feature'})
            vec = {'type' : 'FeatureCollection', 'features' : features}
            return vec, i, j, sample

def dump(features, img_data, sample_name, plot = False):
    if not os.path.exists(os.path.dirname(sample_name)):
        os.makedirs(os.path.dirname(sample_name))

    if plot:
        img_data = img_data.copy()
        for feature in features['features']:
            b = list(map(int, shape(feature['geometry']).bounds))
            cv2.rectangle(img_data, tuple(b[:2]), tuple(b[2:]), (0, 0, 255))
        cv2.imwrite(sample_name + '.jpg', img_data)

    json.dump(features, open(sample_name + '.json', 'w'))

def generate_samples(model, country, threshold, N, ensemble = False):
    '''
    Main function to generate training samples
    '''
    read_cur, write_cur = conn.cursor(), conn.cursor()
    read_cur.execute("""
        SELECT filename, shifted FROM buildings.images
        WHERE project=%s AND (done=false OR done IS NULL)
        ORDER BY random()
        LIMIT 1000
    """, (country,))

    augs = [
        noop,                    # leave image unchanged
        partial(rotate, 180),    # flip it upside down
        mirror,                  # mirror it
        distort,                 # keep dimensions, but distort the color channels
        partial(crop, corner=0), # crop the top left corner and stretch
        partial(crop, corner=1), # crop the top right corner and stretch
        partial(crop, corner=2), # crop the bottom left corner and stretch
        partial(crop, corner=3)  # crop the bottom right corner and stretch
    ]

    if not ensemble:
        augs = [noop]

    TS = datetime.now().isoformat()

    for file, geom in read_cur:
        result = process_file(file, write_cur, augs, model, threshold, ensemble)
        if result:
            features, roff, coff, img_data = result

            geom = wkb.loads(geom, hex=True)
            minlon, minlat, maxlon, maxlat = geom.bounds
            gsd = get_gsd(minlat, 18) # images we've gathered are from zoom level 18 on Bing Maps

            # Compute the lat/lon bounds of the image sample
            cropped_geom = box(
                minlon + coff * gsd,
                minlat + roff * gsd,
                minlon + (coff + img_data.shape[1]) * gsd,
                minlat + (roff + img_data.shape[0]) * gsd
            )

            features['properties'] = {'image' : file, 'roff' : roff, 'coff' : coff}
            sample_name = os.path.join('generated_training_data', TS, 'sample_%d' % N)
            dump(features, img_data, sample_name, plot = True)
            N -= 1
            tqdm.write(str(N))

        write_cur.execute("UPDATE buildings.images SET done=true WHERE filename=%s AND project=%s", (file,country))
        conn.commit()
        if N == 0: return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--gpu', default='0', type=str, help='Which GPU to run on')
    parser.add_argument('--country', required=True, type=str, help='Which country to sample from')
    parser.add_argument('--num', '-n', default=20, type=int, help='Number of samples to generate')
    parser.add_argument('--verbose', '-v', default=False, help='Print progess bar information')
    parser.add_argument('--threshold', '-t', default = None, type=float, help='Confidence threshold')
    # NOTE: As is, this currently seems to not be useful, recommend not using...
    parser.add_argument('--ensemble', action='store_true', help='Run with ensemble agreement')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if not os.path.exists('generated_training_data'):
        os.mkdir('generated_training_data')

    # Retrieve the top performing model from our database
    model, model_id, threshold = get_best_model()
    if args.threshold: threshold = args.threshold

    generate_samples(model, args.country, threshold, args.num, ensemble=args.ensemble)
