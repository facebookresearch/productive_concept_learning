# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import numpy as np
import random
import soundfile as sf
import json
import torch
import logging

from typing import List, Dict

# TODO(ramav): Consolidate these settings into the rest of the code config.
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = '/checkpoint/aszlam/audio_clips/wavs'
INSTRUMENTS = ['trumpet', 'clarinet', 'violin', 'flute', 'oboe', 'saxophone', 'french-horn', 'guitar']
CMAP = {"gray": 0,
        "red": 1,
        "blue": 2,
        "green": 3,
        "brown": 4,
        "purple": 5,
        "cyan": 6,
        "yellow": 7}

NOTES = ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'A5']
NOTES_ALT = ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'A4'] #for french-horn and guitar

#get from metadata?
SHAPEMAP = {"cube":0, "sphere":1, "cylinder":2}
SIZEMAP = {"large":1.0, "small":.3}
L = 40000
TL = 120000
FILTN = 1000
MAX_COMP = 8
NUM_POSITIONS = 8
VOLUMES = len(SIZEMAP)


def filt(x):
    xn = np.linalg.norm(x)
    u = np.fft.fft(x)
    u[FILTN:] = 0
    z = np.real(np.fft.ifft(u))
    zn = np.linalg.norm(z)
    return z*(xn/zn)

MATERIALSMAP  = {"rubber": lambda x: x, "metal": filt} 



def get_bins(obj, metadata):
    dimensions_to_idx = metadata['dimensions_to_idx']
    nbinsx = metadata["location_bins"]["x"]
    nbinsy = metadata["location_bins"]["y"]
    xbin = int(math.floor(obj["pixel_coords"][dimensions_to_idx["x"]] /
                   (metadata["image_size"]["x"] * (1.0 / nbinsx))))
    ybin = int(math.floor(obj["pixel_coords"][dimensions_to_idx["y"]] /
                   (metadata["image_size"]["y"] * (1.0 / nbinsy))))
    return xbin, ybin


class ClevrJsonToSoundTensor(object):
    def __init__(self, metadata_path):
        self.TL = TL
        self.metadata = json.load(open(metadata_path, 'r'))['metadata']
        self.clips = []
        self.clip_info = []
        for fname in os.listdir(SOURCE_DIR):
            q = fname.split('_')
            data, _ = sf.read(os.path.join(SOURCE_DIR, fname))
            if len(data) >= L:
                try:
                    i = INSTRUMENTS.index(q[0])
                    if q[0] == "guitar" or q[0] == "french-horn":
                        pitch = NOTES_ALT.index(q[1])
                    else:
                        pitch = NOTES.index(q[1])
                    self.clips.append(data[:L])
                    self.clip_info.append([i, pitch])
                except:
                    continue
        self.clips_by_prop = {}
        for i in range(len(INSTRUMENTS)):
            self.clips_by_prop[i] = {}
            for j in range(len(NOTES)):
                self.clips_by_prop[i][j] = []
        
        for c in range(len(self.clip_info)):
            i = self.clip_info[c][0]
            pitch = self.clip_info[c][1]
            self.clips_by_prop[i][pitch].append(c)
            
        self.offsets = np.floor((np.linspace(0, TL - L, NUM_POSITIONS))).astype('int64')
        self.masks = [np.ones(L),
                      np.linspace(1, 0, L)**2,
                      np.linspace(0, 1, L)**2]
        
    # TODO(ramav): Add material into the sound creation pipeline.
    def __call__(self, json_objects: List[Dict]):
        out = np.zeros(TL)
        for obj in json_objects:
            instrument = CMAP[obj['color']]
            shift, pitch = get_bins(obj, self.metadata)
            #FIX THE EMPTY clip_by_prop bin!!!!!
            try:
                clipid = random.choice(self.clips_by_prop[instrument][pitch])
            except:
                logging.info('warning: bad translation bc missing pitch, FIXME')
                continue
            c = self.clips[clipid]
            c = MATERIALSMAP[obj['material']](c)
            mask = self.masks[SHAPEMAP[obj['shape']]]
            v = SIZEMAP[obj['size']]
            c = c*mask*v
            offset = self.offsets[shift]
            out[offset:offset + L] = out[offset:offset + L] + c
        return out
            
    def generate_from_json(self, fpath):
        with open(fpath) as j:
            spec = json.load(j)
            return self.__call__(spec["objects"])
                
    def generate_random(self): 
        num_comp = np.random.randint(MAX_COMP)
        clip = np.zeros(TL)
        info = []
        for i in range(num_comp):
            j = np.random.randint(len(self.clips))
            p = np.random.randint(NUM_POSITIONS)
            c = self.clips[j]
            vid = np.random.randint(VOLUMES)
            v = vid/VOLUMES
            mid = np.random.randint(len(self.masks))
            mask = self.masks[mid]
            c = c*mask*v
            offset = self.offsets[p]
            clip[offset:offset + L] = clip[offset:offset + L] + c
            info.append([self.clip_info[j][0],
                         self.clip_info[j][1],
                         p,
                         mid,
                         vid])
        return clip, info



if __name__ == "__main__":
    import visdom
    import json
    vis = visdom.Visdom(server ='http://localhost')
    S = ClevrJsonToSoundTensor('/private/home/aszlam/junk/clevr_typed_fol_properties.json')
    
