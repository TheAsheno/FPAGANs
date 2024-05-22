import os

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class Img_to_zero_center(object):
    def __call__(self, t_img):
        t_img = (t_img - 0.5) * 2
        return t_img

class Reverse_zero_center(object):
    def __call__(self, t_img):
        t_img = t_img / 2 + 0.5
        return t_img