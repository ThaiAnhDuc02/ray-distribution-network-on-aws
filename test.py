# Ray example

import ray
import cv2
import numpy
import time

# @ray.remote
def test(img):
    orb = cv2.AKAZE_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return des

test_remote = ray.remote(test)

if __name__ == "__main__":
    ray.init()
    start_time = time.time()
    
    img = cv2.imread("./tmp/test.jpg")
    futures = [test_remote.remote(img) for i in range(100)]
    ray.get(futures)
    
    print(time.time() - start_time)
