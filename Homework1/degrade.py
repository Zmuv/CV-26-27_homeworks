import cv2
import numpy as np

"""
This script slightly reduces contrast and adds noise to the images
"""


IMAGES_NAMES = ["low_light.png", "outside.png"]


for image_name in IMAGES_NAMES:

    image = cv2.imread(f"./images/{image_name}")

    # reduce contrast around the mean
    alpha = 0.6
    gray = np.full_like(image, int(image.mean()))
    low = cv2.addWeighted(image, alpha, gray, 1 - alpha, 0)

    # Add gaussian noise
    mean = 0
    std = 25
    noise = np.random.normal(mean, std, low.shape)
    noisy_image = np.clip(low.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(f"./images/degraded/{image_name}", noisy_image)
