import cv2
import numpy as np

IMAGES_NAMES = ["low_light.png", "outside.png"]

for image_name in IMAGES_NAMES:
    image_filepath = f"./images/degraded/{image_name}"

    # Read the image
    image = cv2.imread(image_filepath)

    # -- Linear contrast stretching --

    # Convert the image to HLS
    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # RGB VERSION
    # Separate the channels
    b, g, r = cv2.split(image)
    b_f = b.astype(np.float32)
    g_f = g.astype(np.float32)
    r_f = r.astype(np.float32)

    # Run linear contrast stretching formula on each channel
    # ((v - v_min)/(v_max - v_min)) * 255 where v is the value of the channel
    new_r = ((r_f - r_f.min()) / (r_f.max() - r_f.min()) * 255).astype(np.uint8)
    new_g = ((g_f - g_f.min()) / (g_f.max() - g_f.min()) * 255).astype(np.uint8)
    new_b = ((b_f - b_f.min()) / (b_f.max() - b_f.min()) * 255).astype(np.uint8)

    # Merge the channels back
    out_image_rgb = cv2.merge([new_b, new_g, new_r])
    cv2.imwrite(f'./images/output/linear_stretch/rgb_{image_name}', out_image_rgb)

    # LUMINANCE VERSION
    # Separate the channels
    h, l, s = cv2.split(image_hls)

    # Convert to float
    l_f = l.astype(np.float32)

    # Run linear contrast stretching formula on the luminance
    # ((l - l_min)/(l_max - l_min)) * 255
    new_l = ((l_f - l_f.min()) / (l_f.max() - l_f.min()) * 255).astype(np.uint8)

    # Merge the channels back
    out_image_hls = cv2.cvtColor(cv2.merge([h, new_l, s]), cv2.COLOR_HLS2BGR)

    # Save the image so that we can calculate metrics later
    cv2.imwrite(f'./images/output/linear_stretch/hls_{image_name}', out_image_hls)

    # -- Histogram equalization --

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the channels
    h, s, v = cv2.split(image_hsv)

    # Equalize the value channel
    new_v = cv2.equalizeHist(v)

    # Merge the channels and convert back to bgr
    out_image_hsv = cv2.cvtColor(cv2.merge([h, s, new_v]), cv2.COLOR_HSV2BGR)

    # Save the image so that we can calculate metrics later
    cv2.imwrite(f"./images/output/histogram_equalization/{image_name}", out_image_hsv)

    # -- Denoising + Gamma correction --
    # Apply bilater filter
    image_smooth = cv2.bilateralFilter(image, 9, 75, 75)

    # Up the gamma

    # Create a LUT as there are only 256 possible values
    # Gamma correction: gamma < 1 brightens
    gamma = 0.7
    lut = (
        np.array([255 * (i / 255) ** gamma for i in range(256)])
        .clip(0, 255)
        .astype(np.uint8)
    )

    # Each value is given directly by the LUT
    out_image_gamma = cv2.LUT(image_smooth, lut)

    cv2.imwrite(f"./images/output/smoothing/{image_name}", out_image_gamma)
