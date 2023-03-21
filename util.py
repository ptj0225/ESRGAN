import itertools
import cv2
import os

def preprocess_img(dirs):
    for dir in dirs:
        img = cv2.imread(dir, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if h == 128 and w == 128:
            continue
        n_h = int(h / 128)
        n_w = int(w / 128)

        file_name = os.path.split(dir)[-1]
        file_name = file_name.split('.')[-2]

        count = 0
        for x, y in itertools.product(range(n_w), range(n_h)):
            x = int(x * (w/ n_w))
            y = int(y * (h/ n_h))
            patch = img[y: y+128, x: x+128, :]
            cv2.imwrite(filename='./train/' + file_name + '_' + str(count) + '.png', 
                        img=patch)
            count += 1

        os.remove(dir)
