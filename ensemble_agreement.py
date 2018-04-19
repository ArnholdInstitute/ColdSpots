#!/usr/bin/env python

import pdb, psycopg2, pandas, rtree, cv2, os, json, argparse, glob, re
from shapely.geometry import box, mapping, shape

def area(o):
    return (o.x2 - o.x1) * (o.y2 - o.y1)

def jaccard(o1, o2):
    intersection = (min(o1.x2, o2.x2) - max(o1.x1, o2.x1)) * \
        (min(o1.y2, o2.y2) - max(o1.y1, o2.y1))
    union = area(o1) + area(o2) - intersection
    return float(intersection) / float(union)

def agree(df, min_jaccard = 0.4, write_data = False):
    df.sort_values(by='score', inplace=True, ascending=False)
    idx = rtree.index.Index()
    for i, row in df.iterrows():
        idx.insert(i, (row.x1, row.y1, row.x2, row.y2), obj=row)

    groups = []
    processed = set()
    for i, row in df.iterrows():
        if i in processed:
            continue
        processed.add(i)

        group = [row]
        overlap = list(idx.intersection((row.x1, row.y1, row.x2, row.y2), objects=True))
        for item in overlap:
            if jaccard(row, item.object) > min_jaccard and item.id not in processed:
                group.append(item.object)
                processed.add(item.id)
        groups.append(group)

    features = []
    for g in groups:
        if len(g) > 1:
            x1, y1, x2, y2 = pandas.DataFrame(g)[['x1', 'y1', 'x2', 'y2']].mean().round().astype(int)
            features.append({'geometry' : mapping(box(x1, y1, x2, y2)), 'properties' : {}, 'type' : 'Feature'})

    vec = {
        'type' : 'FeatureCollection',
        'features' : features
    }

    if not write_data:
        return vec

    filename = os.path.splitext('_'.join(df.filename[0].split('/')[1:]))[0]
    outimg = 'distilled_data/3band/%s_%d_%d.jpg' % (filename, df.roff[0], df.coff[0])
    outvec = 'distilled_data/vectordata/%s_%d_%d.geojson' % (filename, df.roff[0], df.coff[0])
    outplot = 'distilled_data/plots/%s_%d_%d.jpg' % (filename, df.roff[0], df.coff[0])

    if os.path.exists(outimg) and os.path.exists(outvec):
        return

    if not os.path.exists(os.path.dirname(outimg)):
        os.makedirs(os.path.dirname(outimg))
    if not os.path.exists(os.path.dirname(outvec)):
        os.makedirs(os.path.dirname(outvec))
    if not os.path.exists(os.path.dirname(outplot)):
        os.makedirs(os.path.dirname(outplot))

    x, y = df[['roff', 'coff']].values[0]
    img = cv2.imread(os.path.join('../data', df.filename[0]))[:-25, :, :]  # strip off Bing logo
    img = img[x:x+500, y:y+500, :]

    with_rects = img.copy()
    for feature in features:
        bounds = shape(feature).bounds
        cv2.rectangle(with_rects, tuple(bounds[:2]), tuple(bounds[2:4]), (0,0,255)) 

    cv2.imwrite(outimg, img)
    cv2.imwrite(outplot, with_rects)

    json.dump(vec, open(outvec, 'w'))

def mk_data():
    countries = {}
    verified = json.load(open('../data/train_data_2017-12-12T15:54:00.565852.json'))

    for sample in verified:
        country = re.search('(.*)_sample_\d+.jpg', os.path.basename(sample['image_path'])).group(1)
        if country == 'gtm':
            continue

        if country in countries:
            countries[country].append(sample)
        else:
            countries[country] = [sample]

    files = glob.glob('distilled_data/3band/*.jpg')

    for file in files:
        country = re.search('(.*)_image_.*.jpg', os.path.basename(file)).group(1)

        rects = []
        for feature in json.load(open(file.replace('.jpg', '.geojson').replace('3band', 'vectordata')))['features']:
            geom = shape(feature)
            keys = ['x1', 'y1', 'x2', 'y2']
            rects.append(dict(zip(keys, geom.bounds)))

        sample = {
            'image_path' : os.path.join('../ColdSpots', file),
            'rects' : rects
        }

        if country in countries:
            countries[country].append(sample)
        else:
            countries[country] = [sample]
    
    lens = [len(countries[k]) for k in countries.keys()]
    max_len = max(lens)

    print(json.dumps(dict(zip(countries.keys(), lens)), indent=2))

    for country in countries.keys():
        iters = 0
        samples = countries[country]
        while iters < 3 and len(countries[country]) < max_len:
            countries[country].extend(samples[:(max_len - len(countries[country]))])
            iters += 1

    result = []
    for k in countries.keys():
        result.extend(countries[k])
    json.dump(result, open('../data/distilled_training.json', 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mk_data', default=False, action='store_true', help='Create train/test sets')
    args = parser.parse_args()

    if args.mk_data:
        mk_data()
    else:
        conn = psycopg2.connect(dbname='aigh', host='/tmp')
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS distill.agreed(
                    x1 int, y1 int, x2 int, y2 int, score real, filename text, roff int, coff int
                );
            """)

            cur.execute("""
                SELECT json_agg(boxes.*)
                FROM distill.images JOIN distill.boxes USING (filename)
                GROUP BY images.filename, roff, coff
                HAVING COUNT(*) > 0
            """)

            for info, in cur:
                df = pandas.DataFrame(info)
                agree(df, write_data = True)
