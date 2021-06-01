#!/usr/bin/env python


#   █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
#   █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█

from umucv.stream import autoStream
from umucv.util import putText

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np

from collections import deque
from statistics import mode


#   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░
#   █░▀░█ █▄█ █▄▀ ██▄ █▄▄

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=False)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# Initialize model
model = ResNet9(1,26)

# Load the model weights
if torch.cuda.is_available():
    model.load_state_dict(torch.load('my_model.pth'))
else:
    model.load_state_dict(torch.load('my_model.pth', map_location='cpu'))
    

#   ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
#   █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█

def predict_sign(sign):
    sign = cv.resize(sign, (28,28))
    sign = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)
    sign = np.array(sign)
    sign = torch.from_numpy(sign).type(torch.FloatTensor)
    sign = torch.reshape(sign, (1, 28, 28))

    input = sign.unsqueeze(0)
    output = F.softmax(model(input), dim=1)#output = model(input)
    _, preds = torch.max(output, dim=1)

    predicted_sign = chr(preds[0].item() + 65)
    predicted_probability = round(output[0][preds].item() * 100, 2)

    return predicted_sign, predicted_probability

def nothing(x):
    pass

mainWindowName = 'ASL classifier'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)
cv.createTrackbar('umbral', mainWindowName, 100, 255, nothing)
  
x1, y1, wh = 300, 100, 28*10
x2, y2 = x1+wh, y1+wh
history = deque(maxlen=6)
bg = []
kernel = np.ones((3,3),np.uint8)

for key, frame in autoStream():

    # Flip frame horizontally so the app is easier to use
    frame = cv.flip(frame, 1)

    # Extract information from ROI and background
    roi = frame[y1:y2, x1:x2]
    cv.imshow('ROI', roi)
    
    # Generate mask based on difference threshold
    if (bg == []) or (key == ord('q')):
        bg = roi
    
    mask = np.sum(cv.absdiff(bg,roi), axis=2) > cv.getTrackbarPos('umbral', mainWindowName)
    mask = np.uint8(mask)*255
    mask = cv.erode(mask,kernel,iterations = 1)
    mask = cv.dilate(mask,kernel,iterations = 1)
    mask = cv.medianBlur(mask, 3)
    cv.imshow('Mask', mask)

    # Extract hand with mask
    hand = roi.copy()
    hand[mask == 0] = 255
    cv.imshow('Masked Hand', hand)

    # Predict the sign
    pred_sign, pred_prob = predict_sign(hand)

    # Show prediction based on a k-frame history and high probability
    if pred_prob > 95:
        history.append(pred_sign)
        prediction = mode(history)
        probability = pred_prob
        print(history)
    
    if len(history) > 0 and history.count(history[0]) > len(history)/2:
        text = prediction+' ('+str(probability)+'%)'
    else:
        text = '?'
    
    putText(frame, 'Prediction: '+text, orig=(x1,y1-8))
    cv.rectangle(frame, (x1,y1-1), (x2,y2-1), color=(0,255,255), thickness=2)
    cv.imshow(mainWindowName, frame)

cv.destroyAllWindows()