from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import time
import warnings

warnings.simplefilter('ignore')

config_path='cfg/yolov3.cfg'
weights_path='yolov3.weights'
class_path='data/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4
# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
print("model loaded")


def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms=transforms.Compose([transforms.Resize((imh,imw)),
         transforms.Pad((max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture
import cv2
from sort import *

vid = cv2.VideoCapture("short_traffic_demo.mp4")

fps = int(vid.get(cv2.CAP_PROP_FPS))
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

mot_tracker = Sort()
tests = []
check_id = deque()
centers = None
setcolor = (0,102,153)

count = 0

while(vid.isOpened()):
    ret, frame = vid.read()

    if type(frame) == type(None): break

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    img = np.array(pilimg)
    mot_tracker.img_shape = [img.shape[0], img.shape[1]]
    mot_tracker.pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    mot_tracker.pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    mot_tracker.unpad_h = img_size - mot_tracker.pad_y
    mot_tracker.unpad_w = img_size - mot_tracker.pad_x

    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        for i, trk in enumerate(mot_tracker.trackers):
            color = colors[int(trk.id+1) % len(colors)]
            color = [i * 255 for i in color]
            trk.color = color

            cls = classes[int(trk.objclass)]

            d = trk.get_state()[0]
            box_w = int(((d[2] - d[0]) / mot_tracker.unpad_w) * img.shape[1])
            box_h = int(((d[3] - d[1]) / mot_tracker.unpad_h) * img.shape[0])
            x1 = int(((d[0] - mot_tracker.pad_x / 2) / mot_tracker.unpad_w) * img.shape[1])
            y1 = int(((d[1] - mot_tracker.pad_y / 2) / mot_tracker.unpad_h) * img.shape[0])

            for i in classes:
                if cls == i:
                    count += 1
            
            xbox_size = x1 + box_w
            ybox_size = y1 + box_h
            
            bbox_tag = cls + " " + str(xbox_size) + " x " + str(ybox_size)

            cv2.rectangle(frame, (x1, y1), (xbox_size, ybox_size), trk.color, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), trk.color, -1)
            cv2.putText(frame, bbx_tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, setcolor, 1)
            print("objects = ", count, "box size = ", xbox_size, ybox_size)

    text = "objects = " + ('%.d' % count)
    cv2.putText(frame, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    count = 0

    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #cv2.imshow(flscr, frame)
    out.write(frame)
    #print("video saved!")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        print('deleted')

out.release()
vid.release()
print("done! video saved")
