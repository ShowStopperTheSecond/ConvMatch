import numpy as np
import argparse
import os
import glob
import cv2
import h5py
from tqdm import tqdm
import subprocess
import sys
import logging
import argparse
import numpy as np
from copy import deepcopy

def clone_repo(repo_url, local_path):
    subprocess.run(['git', 'clone', repo_url, local_path])




def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='../raw_data/yfcc100m/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, default='*/*/images/*.jpg',
  help='Glob match if directory of images is specified (default: \'*/images/*.jpg\').')
parser.add_argument('--num_kp', type=int, default='2000',
  help='keypoint number, default:2000')
parser.add_argument('--suffix', type=str, default='sift-2000',
  help='suffix of filename, default:sift-2000')
parser.add_argument("--feature_extractor", type=str, default="sift")


class ExtractALIKED(object):
  def __init__(self):
    clone_repo("https://github.com/Shiaoming/ALIKED.git", "/tmp/aliked")
    subprocess.run("cd /tmp/aliked/custom_ops; bash /tmp/aliked/custom_ops/build.sh")
    sys.path.append("/tmp/aliked/")
    from nets.aliked import ALIKED
    self.feature_extractor = ALIKED

  def run(self, img_path):
    img = cv2.imread(img_path)
    pred_ref = model.run(img)
    kpts_ref = pred_ref['keypoints']
    desc_ref = pred_ref['descriptors']
    return kp, desc
    




class ExtractSIFT(object):
  def __init__(self):
    self.feature_extractor = cv2.SIFT_create()

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.feature_extractor.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc


class ExtractORB(object):
  def __init__(self):
    self.feature_extractor = cv2.ORB_create()

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.feature_extractor.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

class ExtractBRISK(object):
  def __init__(self):
    self.feature_extractor = cv2.BRISK_create()

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.feature_extractor.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

class ExtractKAZE(object):
  def __init__(self):
    self.feature_extractor = cv2.KAZE_create()

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.feature_extractor.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc

class ExtractAKAZE(object):
  def __init__(self):
    self.feature_extractor = cv2.AKAZE_create()

  def run(self, img_path):
    img = cv2.imread(img_path)
    cv_kp, desc = self.feature_extractor.detectAndCompute(img, None)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc




def write_feature(pts, desc, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors"][:] = desc


if __name__ == "__main__":



  opt = parser.parse_args()
  if opt.feature_extractor == "sift":
    detector = ExtractSIFT()
  elif  opt.feature_extractor == "orb":
    detector = ExtractORB()
  elif  opt.feature_extractor == "kaze":
    detector = ExtractKAZE()
  elif  opt.feature_extractor == "akaze":
    detector = ExtractAKAZE()
  elif  opt.feature_extractor == "brisk":
    detector = ExtractBRISK()
  else:
    print(f"{opt.feature_extractor}: Feature extractor doesn't exist")

  # get image lists
  search = os.path.join(opt.input_path, opt.img_glob)
  listing = glob.glob(search)

  for img_path in tqdm(listing):
    kp, desc = detector.run(img_path)
    save_path = img_path+'.'+opt.suffix+'.hdf5'
    write_feature(kp, desc, save_path)
