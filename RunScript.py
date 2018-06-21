import os
from TensorBox import train as TY
import json

with open('hypes.json') as json_file:
        H = json.load(json_file)


#os.environ['CUDA_VISIBLE_DEVICES'] = str(int(1))

#H["data"]['train_idl'] = '../ColdSpots/data/modified_training_data.json'
#H["data"]['root_dir'] = ''
TY.train(H,[])
