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
import multiprocessing as mp
from typing import Tuple, Any


def clone_repo(repo_url, local_path):
    subprocess.run(['git', 'clone', repo_url, local_path])


def safe_double_desc(img, featureExtractor1, featureExtractor2, timeout: int = 30):
    """
    Safely process a single image pair with feature extractors in a separate process.
    
    Args:
        img:  image
        featureExtractor1: First feature extractor
        featureExtractor2: Second feature extractor
        timeout: Maximum time in seconds to wait
        
    Returns:
        Tuple of (results, error_message)
        - If successful, results contains (kp1_first, desc1_first, ...) and error_message is None
        - If failed, results is None and error_message contains the error
    """
    def worker(img, featureExtractor1, featureExtractor2, return_dict):
        try:
            kp_first, desc_first = featureExtractor1.detectAndCompute(img, None)
            kp_second, desc_second = featureExtractor2.compute(img, kp_first)
            k = np.array([k.pt for k in kp_first])
            kk = np.array([k.pt for k in kp_second])
            dist = np.sum(np.abs(k[None, :, :] - kk[:, None, :]), -1)
            same = np.argwhere(dist == 0)
            keypoints = np.take(kp_first, same[:,1],0)
            desc_1st = desc_first[ same[:, 1],:]
            desc_2nd = desc_second[ same[:, 0],:]
            keypoints = cv2.KeyPoint_convert(keypoints)
            return_dict['result'] = (keypoints, desc_1st, desc_2nd)

        except Exception as e:
            return_dict['error'] = str(e)

    # Create a manager to share results between processes
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Create and start the process
    process = mp.Process(
        target=worker,
        args=(img, featureExtractor1, featureExtractor2, return_dict)
    )
    process.start()
    process.join(timeout)
    
    # Handle various failure cases
    if process.is_alive():
        process.terminate()
        process.join()
        return None, "Process timed out"
    
    if process.exitcode != 0:
        return None, f"Process crashed with exit code {process.exitcode}"
        
    if 'error' in return_dict:
        return None, return_dict['error']
        
    if 'result' in return_dict:
        return return_dict['result'], None
        
    return None, "Unknown error occurred"

class DoubleDesc(object):
  def __init__(self, featureExtractor1, featureExtractor2):
    self.featureExtractor1 = featureExtractor1
    self.featureExtractor2 = featureExtractor2

  def run(self, img_path):
    img = cv2.imread(img_path)
    result, error = safe_double_desc(img, self.featureExtractor1, self.featureExtractor2)
    if error is None:
      cv_kp , desc1, desc2 = result
    else:
      print("error happened in worker")
      print(error)

      return None, None, None
    cv_kp = cv2.KeyPoint_convert(cv_kp)
    kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4
    return kp, desc1, desc2





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
    current_directory = os.getcwd()
    os.chdir("/tmp/aliked/custom_ops")
    subprocess.run(["bash", "/tmp/aliked/custom_ops/build.sh"])
    sys.path.append("/tmp/aliked/")
    from nets.aliked import ALIKED
    self.feature_extractor = ALIKED()
    os.chdir(current_directory)

  def run(self, img_path):
    img = cv2.imread(img_path)
    pred_ref = self.feature_extractor.run(img)
    kpts_ref = pred_ref['keypoints']
    desc_ref = pred_ref['descriptors']
    return kpts_ref, desc_ref
    

class ExtractEnhancedALIKED(object):
  def __init__(self, n_descriptors=2):
    clone_repo("https://github.com/ShowStopperTheSecond/EnhancedALIKED", "/tmp/EnhancedALIKED")
    current_directory = os.getcwd()
    if not os.path.exists("/tmp/EnhancedALIKED/build"):
      os.chdir("/tmp/EnhancedALIKED/custom_ops")
      subprocess.run(["bash", "/tmp/EnhancedALIKED/custom_ops/build.sh"])
    sys.path.append("/tmp/EnhancedALIKED/")
    os.chdir("/tmp/EnhancedALIKED/")
    from nets.aliked import EnhancedALIKED
    self.feature_extractor = EnhancedALIKED(load_pretrained=True, n_limit=100000,)
    os.chdir(current_directory)
    self.n_descriptors = n_descriptors

  def run(self, img_path):
    img = cv2.imread(img_path)
    pred_ref = self.feature_extractor.run(img)
    kpts_ref = pred_ref['keypoints']
    desc_ref = pred_ref['descriptors'][: self.n_descriptors]
    return kpts_ref, desc_ref
    




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

def write_feature_double_desc(pts, desc1, desc2, filename):
  with h5py.File(filename, "w") as ifp:
      ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
      ifp.create_dataset('descriptors1', desc1.shape, dtype=np.float32)
      ifp.create_dataset('descriptors2', desc2.shape, dtype=np.float32)
      ifp["keypoints"][:] = pts
      ifp["descriptors1"][:] = desc1
      ifp["descriptors2"][:] = desc2


if __name__ == "__main__":


  single_descs = ["sift", "orb", "kaze", "akaze", "brisk", "aliked"]
  double_descs = ["sift_brisk", "brisk_sift", "orb_brisk", "orb_sift", "double_aliked", "triple_aliked" ]

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
  elif opt.feature_extractor == "aliked":
    detector = ExtractALIKED()
  elif opt.feature_extractor == "sift_brisk":
    detector = DoubleDesc(cv2.SIFT_create(), cv2.BRISK_create())
  elif opt.feature_extractor == "brisk_sift":
    detector = DoubleDesc(cv2.BRISK_create(), cv2.SIFT_create())
  elif opt.feature_extractor == "orb_brisk":
    detector = DoubleDesc(cv2.ORB_create(), cv2.BRISK_create())
  elif opt.feature_extractor == "orb_sift":
    detector = DoubleDesc(cv2.ORB_create(), cv2.SIFT_create())
  elif opt.feature_extractor == "double_aliked":
    detector = ExtractEnhancedALIKED()
  elif opt.feature_extractor == "triple_aliked":
    detector = ExtractEnhancedALIKED(n_descriptors=3)
  else:
    print(f"{opt.feature_extractor}: Feature extractor doesn't exist")


  # get image lists
  search = os.path.join(opt.input_path, opt.img_glob)
  listing = glob.glob(search)

  if opt.feature_extractor in single_descs:
    for img_path in tqdm(listing):
      kp, desc = detector.run(img_path)
      save_path = img_path+'.'+opt.suffix+'.hdf5'
      write_feature(kp, desc, save_path)

  elif opt.feature_extractor in double_descs:
    for img_path in tqdm(listing):
      kp, desc1, desc2 = detector.run(img_path)
      if kp is None:
        print("couldn't extract features for this image")
        continue
      save_path = img_path+'.'+opt.suffix+'.hdf5'
      write_feature_double_desc(kp, desc1, desc2, save_path)


