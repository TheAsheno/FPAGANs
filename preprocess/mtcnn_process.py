from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
import os
from tqdm import tqdm
from skimage import transform as trans

root_data_dir = "dataset/CACD2000/image"
origin_image_dir = os.path.join(root_data_dir, "origin")
store_image_dir = os.path.join(root_data_dir, "expriment")

if not os.path.exists(store_image_dir):
    os.makedirs(store_image_dir)

src = np.array([
 [30.2946, 51.6963],
 [65.5318, 51.5014],
 [48.0252, 71.7366],
 [33.5493, 92.3655],
 [62.7299, 92.2041] ], dtype=np.float32 )

threshold = [0.6,0.7,0.9]
factor = 0.85
minSize = 20
imgSize = [120, 100]

detector = MTCNN(steps_threshold=threshold, scale_factor=factor, min_face_size=minSize)

keypoint_list = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']

for i, filename in enumerate(tqdm(os.listdir(origin_image_dir))):
    if i < 75393:
        continue

    dst = []
    filepath = os.path.join(origin_image_dir, filename)
    storepath = os.path.join(store_image_dir, filename)

    with Image.open(filepath) as img:
        npimage = np.array(img)

    dictface_list = detector.detect_faces(npimage)

    if len(dictface_list) > 1:
        boxs = [dictface['box'] for dictface in dictface_list]
        center = np.array(npimage.shape[:2]) / 2
        boxs = np.array(boxs)
        face_center = boxs[:, :2] + boxs[:, 2:] / 2
        distance = np.sqrt(np.sum(np.square(face_center - center), axis=1))
        min_id = np.argmin(distance)
        dictface = dictface_list[min_id]
    elif dictface_list:
        dictface = dictface_list[0]
    else:
        continue

    face_keypoint = dictface['keypoints']
    dst = [face_keypoint[keypoint] for keypoint in keypoint_list]
    dst = np.array(dst).astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    warped = cv2.warpAffine(npimage, M, (imgSize[1], imgSize[0]), borderValue=0.0)
    warped = cv2.resize(warped, (400, 400))

    with Image.fromarray(warped.astype(np.uint8)) as img:
        img.save(storepath)