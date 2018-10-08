#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 12:28:36 2018

@author: uidaho
"""
import cv2


img=cv2.imread('TCGA-RD-A8N9-01A-01-TS1_b.bmp',cv2.IMREAD_UNCHANGED)
cv2.imwrite('2.tif',img)    