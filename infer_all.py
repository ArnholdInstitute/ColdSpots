#!/usr/bin/env python

import os, json, subprocess, cv2, argparse, pdb, rtree, numpy as np, psycopg2, md5, math
from shapely.geometry import MultiPolygon, box, shape
from multiprocessing import Queue, Process
from Dataset import InferenceGenerator
from datetime import datetime
from geopy.distance import VincentyDistance
from db import db_args, conn
from model import get_best_model

def gsd(lat, zoom):
    """
    Computes the Ground Sample Distance (GSD).  More details can be found
    here: https://msdn.microsoft.com/en-us/library/bb259689.aspx

    Args:
        lat : float - latitude of the GSD of interest
        zoom : int - zoom level (WMTS)
    """
    return (math.cos(lat * math.pi / 180) * 2 * math.pi * 6378137) / (256 * 2**zoom)

def raster_to_proj(x, y, img_geom, ref_point):
    (lon,), (lat,) = img_geom.centroid.xy
    return(
        VincentyDistance(meters=gsd(lat, 18) * x).destination(point=ref_point, bearing=90).longitude,
        VincentyDistance(meters=gsd(lat, 18) * y).destination(point=ref_point, bearing=180).latitude
    )

def process_results(queue, args):
    conn = psycopg2.connect(**db_args)
    cur = conn.cursor()
    while True:
        item = queue.get()
        if item is None:
            return

        boxes, meta, VERSION = item

        (roff, coff, filename, valid_geom, done, height, width, img_geom) = meta
        img_geom = shape(img_geom)

        bounds = img_geom.bounds
        ref_point = (bounds[3], bounds[0]) # top left corner

        for _, r in boxes.iterrows():
            minx, miny = raster_to_proj(r.x1 + coff, r.y1 + roff, img_geom, ref_point)
            maxx, maxy = raster_to_proj(r.x2 + coff, r.y2 + roff, img_geom, ref_point)
            building = box(minx, miny, maxx, maxy)

            cur.execute("""
                INSERT INTO buildings.buildings (filename, minx, miny, maxx, maxy, roff, coff, score, project, ts, version, geom)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::uuid, ST_GeomFromText(%s, 4326))
            """, (filename, int(r.x1), int(r.y1), int(r.x2), int(r.y2), roff, coff, r.score, args.country, args.ts, VERSION, building.wkt))
        
        if done:
            cur.execute("UPDATE buildings.images SET last_tested=%s WHERE project=%s AND filename=%s", (args.ts, args.country, filename))
            conn.commit()
            print('Committed image: %s' % filename)

def infer_all(model, args):
    queue = Queue()
    args.ts = datetime.now().isoformat()

    ps = []
    for _ in range(4):
        processor = Process(target=process_results, args=(queue, args))
        processor.start()
        ps.append(processor)

    area_to_cover = None
    if args.boundary:
        area_to_cover = shape(json.load(open(args.boundary)))

    generator = InferenceGenerator(conn, args.country, area_to_cover=area_to_cover, data_dir=args.data_dir, threads=8)
    for orig_img, meta in generator:
        result = model.predict_image(orig_img, args.threshold, eval_mode=False)
        queue.put((result, meta, args.model_version))

    for p in ps:
        queue.put(None)
        p.join()

def main():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--gpu', default=0, type=int, help='Which GPU to run on')
    parser.add_argument('--boundary', default=None, type=str, help='Path to GeoJSON file describing boundary to infer')
    parser.add_argument('--country', required=True, type=str, help='Which country to infer')
    parser.add_argument('--data_dir', default='../data', type=str, help='Base directory containing images')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    model, model_version, threshold = get_best_model()
    args.model_version = model_version
    args.threshold = threshold

    infer_all(model, args)

 
if __name__ == '__main__':
    main()