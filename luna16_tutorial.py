import os
import csv
import SimpleITK as sitk
import numpy as np
from PIL import Image
import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing


def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxelToWorldCoord(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord

def seq(start, stop, step=1):
    n = int(round((stop - start)/float(step)))
    if n > 1:
        return([start + step*i for i in range(n+1)])
    else:
        return([])

def normalizePlanes(npzarray):
    maxHU = 400.
    minHU = -1000.

    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray > 1] = 1.
    npzarray[npzarray < 0] = 0.
    return npzarray


img_path = '/Volumes/Hyacinth/Dropbox/Dropbox (123)/TUTORIAL/data/1.3.6.1.4.1.14519.5.2.1.6279.6001.148447286464082095534651426689.mhd'
cand_path = '/Volumes/Hyacinth/Dropbox/Dropbox (123)/TUTORIAL/data/candidates.csv'

# load image
numpyImage, numpyOrigin, numpySpacing = load_itk_image(img_path)
print(numpyImage.shape)
print('Origin is: ')
print(numpyOrigin)
print(numpySpacing)

# load candidates
cands = readCSV(cand_path)
print(cands)

# get candidates
for cand in cands[1:]:
    worldCoord = np.asarray([float(cand[3]), float(cand[2]), float(cand[1])])
    voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    voxelWidth = 65
    coord_start = [0, 0, 0]
    coord_end = [0, 0, 0]
    slice = int(voxelCoord[0])
    coord_start[1] = int(voxelCoord[1] - voxelWidth / 2.0)
    coord_end[1] = int(voxelCoord[1])  # +voxelWidth/2.0)/2.0)
    coord_start[2] = int(voxelCoord[2] - voxelWidth / 2.0)
    coord_end[2] = int(voxelCoord[2] + voxelWidth / 2.0)
    patch = numpyImage[slice, coord_start[1]:coord_end[1], coord_start[2]:coord_end[2]]

    patch = normalizePlanes(patch)
    print('data')
    print(worldCoord)
    print(voxelCoord)
    print(patch)
    outputDir = 'patches/'
    plt.imshow(patch, cmap='gray')
    plt.show()
    # Image.fromarray(patch*255).convert('L').save(os.path.join(outputDir, 'patch_' + str(worldCoord[0]) + '_' + str(worldCoord[1]) + '_' + str(worldCoord[2]) + '.tiff'))
