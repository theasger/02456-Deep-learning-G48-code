#!/zhome/7c/0/155777/anaconda3/envs/ml/bin/python
import numpy as np
import cv2

def draw_masks(image, pred_masks, colors = [(0,0,0),(255,0,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255)]) :
  masked_image = image.copy()

  masks_generated=[pred_masks == c for c in range(len(colors))]
  labels = [[i] for i in range(len(colors))]

  for i in range(len(masks_generated)) :
    masked_image = np.where(np.repeat(masks_generated[i][:, :, np.newaxis], 3, axis=2),
                            np.asarray(colors[int(labels[i][-1])], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)

  return cv2.addWeighted(image.astype(np.uint8), 0.3, masked_image, 0.7, 0)