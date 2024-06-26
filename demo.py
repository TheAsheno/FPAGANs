import torchvision
from model.FPAGANs import FPAGANs
from utils.io import Img_to_zero_center,Reverse_zero_center
from mtcnn.mtcnn import MTCNN
from PIL import Image
import numpy as np
import cv2
from skimage import transform as trans
from torchvision.utils import save_image
import os
import torch

class Demo:
    def __init__(self, generator_state_pth):
        self.model = FPAGANs(modelpath=generator_state_pth)

    def mtcnn_align(self,image):
        dst = []
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        threshold = [0.6, 0.7, 0.9]
        factor = 0.85
        minSize = 20
        imgSize = [120, 100]
        detector = MTCNN(steps_threshold=threshold, scale_factor=factor, min_face_size=minSize)
        keypoint_list = ['left_eye','right_eye','nose','mouth_left','mouth_right']


        npimage = np.array(image)
        dictface_list = detector.detect_faces(npimage)  # if more than one face is detected, [0] means choose the first face

        if len(dictface_list) > 1:
            boxs = []
            for dictface in dictface_list:
                boxs.append(dictface['box'])
            center = np.array(npimage.shape[:2]) / 2
            boxs = np.array(boxs)
            face_center_y = boxs[:, 0] + boxs[:, 2] / 2
            face_center_x = boxs[:, 1] + boxs[:, 3] / 2
            face_center = np.column_stack((np.array(face_center_x), np.array(face_center_y)))
            distance = np.sqrt(np.sum(np.square(face_center - center), axis=1))
            min_id = np.argmin(distance)
            dictface = dictface_list[min_id]
        else:
            if len(dictface_list) == 0:
                return image
            else:
                dictface = dictface_list[0]
        face_keypoint = dictface['keypoints']
        for keypoint in keypoint_list:
            dst.append(face_keypoint[keypoint])
        dst = np.array(dst).astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        warped = cv2.warpAffine(npimage, M, (imgSize[1], imgSize[0]), borderValue=0.0)
        warped = cv2.resize(warped, (200, 200))
        return Image.fromarray(warped.astype(np.uint8))


    def demo(self, image):
        image = self.mtcnn_align(image)
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128,128)),
            torchvision.transforms.ToTensor(),
            Img_to_zero_center()
        ])
        label_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        image = transforms(image).unsqueeze(0)
        imgs = image.cuda()
        for age in range(5):
            full_one = np.ones((128, 128), dtype=np.float32)
            full_zero = np.zeros((128, 128, 5), dtype=np.float32)
            full_zero[:, :, age] = full_one
            label = label_transforms(full_zero).unsqueeze(0)

            img = image.cuda()
            lbl = label.cuda()
            self.model.generator.cuda()

            res = self.model.test_generate(img,lbl)
            imgs = torch.cat((imgs, res), 0)
        save_image(Reverse_zero_center()(imgs), fp=os.path.join('./test/',"res.jpg"))

if __name__ == '__main__':
    D=Demo("checkpoint\\FPAGANS\\2024-05-17_12-56-32\\saved_parameters\\gepoch_5_iter_6500.pth")
    img=Image.open("./test/face14.jpg")
    D.demo(img)
