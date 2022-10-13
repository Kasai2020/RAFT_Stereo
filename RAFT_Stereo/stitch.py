import cv2
import numpy as np

# load camera matrix
rgb = cv2.imread(f"datasets/realsense_infrared_test_data/{0}/laser_on_camera_1.png")

disp = cv2.imread(f"demo_output/{0}.png")
disp = disp[8:-8]

print(np.shape(rgb))
print(np.shape(disp))

im_v = cv2.hconcat([rgb, disp])

img_tot = im_v

for i in range(1,10):
    # load camera matrix
    rgb = cv2.imread(f"datasets/realsense_infrared_test_data/{i}/laser_on_camera_1.png")

    disp = cv2.imread(f"demo_output/{i}.png")
    disp = disp[8:-8]

    print(np.shape(rgb))
    print(np.shape(disp))

    img_tot = cv2.vconcat([img_tot, cv2.hconcat([rgb, disp])])

cv2.imwrite("total_laser.png", img_tot)





