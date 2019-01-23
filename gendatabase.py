"""
this script takes a set of folders inside the directory 'path', 
each folder containing the examples (images) for each class, 
and generate a single h5 file with all the database.
"""

import numpy as np
import os
from PIL import Image
import h5py

path="tropical-fruits"

# variable initialization
d = []
fnames = []
fpaths = []
labels = []
classno = []
classes = set()
j = 0

# recorrer archivos
for (dirpath, dirnames, filenames) in os.walk(path):
    if len(filenames)>0 and dirpath.count(os.sep)==1:
        parts = dirpath.split(os.sep)
        classname = parts[1][1:]
        #print(fnames)
        fnames = fnames+filenames
        labels = labels+([classname] * len(filenames))
        classno = classno+([j] * len(filenames))
        fpaths = fpaths+([dirpath] * len(filenames))
        classes.add(classname)
        j = j+1

num_classes = len(classes)
nb_classes = num_classes
print ("Number of classes: "+str(num_classes))

num_samples = np.size(fnames)
print ("Number of samples: "+str(num_samples))

# open one image to get size
im1 = np.array(Image.open(fpaths[0] + "/" + fnames[0])) 
m,n = im1.shape[0:2] # get the size of the images

print ("Original Image size: "+str(m)+'x'+str(n))

# downsampling
downsample = 16
m = int(m/downsample)
n = int(n/downsample)

print ("Downsampled Image size: "+str(m)+'x'+str(n))

imnbr = len(fnames) # get the number of images

X = np.array([np.array(Image.open(fpaths[j] + "/" +fnames[j]).resize((n,m), Image.ANTIALIAS))#.flatten()
              for j in range(0,num_samples)],'f')

y = np.array(classno,dtype = int)

print ('Saving dataset')
filename = 'fruits_downsampled_' + str(downsample)
f = h5py.File(filename +'.h5', 'w')
# Creating dataset to store data
f.create_dataset('data', data = X, dtype='f')
# Creating dataset to store labels
f.create_dataset('label', data = y, dtype='i')
f.close()