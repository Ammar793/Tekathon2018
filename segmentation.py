import cv2
import numpy as np

def check_if_item_present(file):
    final_answer = False

    imag = './train/train/' + file

    img = cv2.imread(imag)

    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # print(img2)
    h, w = img1.shape

    # print(h)
    # print(w)

    #	keep_going = True
    #	c =0
    #	while(keep_going):

    #		img1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    c2 = 0
    do_left = True
    while (do_left):
        left = np.zeros((h, 5 * (c2 + 1), 3), dtype=np.uint8)
        left2 = np.zeros((h, 5 * (c2 + 1), 3), dtype=np.uint8)
        img_remain = np.zeros((h, w - 5 * (c2 + 1), 3), dtype=np.uint8)
        img_remain2 = np.zeros((h, w - 5 * (c2 + 1), 3), dtype=np.uint8)

        h_l, w_l, c_l = left.shape
        h_i, w_i, c_i = img_remain.shape

        for j in range(0, len(left)):
            for k in range(0, len(left[0])):
                left[j, k] = img[j, k]
                left2[j, k] = img2[j, k]

        for j in range(0, len(img_remain)):
            for k in range(0, len(img_remain[0])):
                img_remain[j, k] = img[j, k + 5 * (c2 + 1)]
                img_remain2[j, k] = img2[j, k + 5 * (c2 + 1)]

        edges_left = cv2.Canny(left, 100, 200)
        edges_remain = cv2.Canny(img_remain, 100, 200)

        ed_left = 0
        ed_remain = 0
        for j in range(0, len(left)):
            for k in range(0, len(left[0])):

                if (edges_left[j, k] == 255):
                    ed_left += 1

        for j in range(0, len(img_remain)):
            for k in range(0, len(img_remain[0])):
                if (edges_remain[j, k] == 255):
                    ed_remain += 1

        average_intensity_left = np.average(np.average(left2, axis=0), axis=0)
        average_intensity_remain = np.average(np.average(img_remain2, axis=0), axis=0)

        ed_left = np.round(ed_left / (h_l * w_l), 2)
        ed_remain = np.round(ed_remain / (h_i * w_i), 2)

        average_intensity_left = np.round(average_intensity_left, 2)
        average_intensity_remain = np.round(average_intensity_remain, 2)

        # print(ed_left)
        # print(ed_remain)
        ##print(average_intensity_left)
        ##print(average_intensity_remain)
        r = abs(int(average_intensity_left[0]) - int(average_intensity_remain[0]))
        g = abs(int(average_intensity_left[1]) - int(average_intensity_remain[1]))
        b = abs(int(average_intensity_left[2]) - int(average_intensity_remain[2]))

        # print(r)
        # print(g)
        # print(b)

        ed_diff = abs(ed_left - ed_remain)

        # print(ed_diff)

        if (r < 1 or g < 1 or b < 1 or ed_diff < 0.02):
            do_left = False
            final_answer = True

        if ( c2 > 20):
            de_left = False
        # print("done")

        c2 += 1

    return final_answer
    #l = 5 * c2