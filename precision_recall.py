#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import pandas, json, pdb, numpy as np, argparse, sys, os, cv2, glob
import matplotlib.pyplot as plt

def get_best(annos, box):
    ovmax = -float('inf')
    best_rect = None
    for r in annos['rects']:
        if 'taken' in r and r['taken']:
            continue
        bi = list(map(max, zip([r['x1'], r['y1']], box[:2]))) + list(map(min, zip([r['x2'], r['y2']], box[2:4])))
        iw=bi[2]-bi[0]+1;
        ih=bi[3]-bi[1]+1;
        if iw > 0 and ih > 0:
            ua=(r['x2']-r['x1']+1)*(r['y2']-r['y1']+1)+ \
               (box[2]-box[0]+1)*(box[3]-box[1]+1)- \
               iw*ih;
            ov = iw*ih/ua
            if ov > ovmax:
                ovmax = ov
                best_rect = r
    if best_rect:
        best_rect['taken'] = True
    return best_rect

def precision_recall(gt, predictions):
    predictions = predictions[['x1', 'y1', 'x2', 'y2', 'score', 'image_id']].values
    predictions = predictions[(-predictions[:, 4]).argsort()]

    tp, fp = np.zeros(len(predictions)), np.zeros(len(predictions))
    for i, (x1, y1, x2, y2, score, image_id) in enumerate(predictions):
        best = get_best(gt[int(image_id)], (x1, y1, x2, y2))
        if best:
            tp[i] = 1
        else:
            fp[i] = 1

    npos = sum(map(lambda x: len(x['rects']), gt))

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp/npos
    prec = tp / (fp+tp)

    return rec, prec

if __name__ == '__main__':
    dflt_val = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/val_data.json')
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', nargs='+', help='Detection csv files')
    parser.add_argument('--names', nargs='+', help='Titles')
    parser.add_argument('--val_data', default=dflt_val, help='Validation data')
    args = parser.parse_args()

    if len(args.predictions) != len(args.names):
        args.names = list(map(lambda x: os.path.basename(x), args.predictions))

    plt.gca().set_prop_cycle('color', ['red', 'green', 'blue', 'yellow'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    for predictions, name in zip(args.predictions, args.names):
        gt = json.load(open(args.val_data))
        df = pandas.read_csv(predictions)
        rec, prec = precision_recall(gt, df)
        ax.plot(rec, prec)

        i = np.argmax(rec + prec)
        f1 = 2 * (prec[i] * rec[i]) / (prec[i] + rec[i])
        print('%s: precision = %f, recall = %f, F1 = %f' % (name, prec[i], rec[i], f1))
    
    plt.legend(args.names, loc='lower left')


    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.savefig('./precision_recall.png')
    plt.close(fig)
















