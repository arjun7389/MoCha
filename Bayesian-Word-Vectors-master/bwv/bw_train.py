# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:11:24 2019

@author: Arjun
"""

import numpy as np
import pandas as pd
import re, random
from tqdm import tqdm
import time



from bwv import *


raw_text = ''
with open('data/frankenstein.txt',encoding="utf8") as f:
    for line in f:
        if 'End of the Project Gutenberg' in line: break
        raw_text += line
        
        
bwv = BWV([raw_text], # the text should come in the form of a list of documents
          m=50, # dimensionality of word embeddings
          tau=1.0, # prior precision
          gamma=0.7, # decay
          n_without_stochastic_update=5,
          vocab_size=20000, 
          sample=0.001)        

epochs = 10
for i in range(epochs):
    bwv.train()