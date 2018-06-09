import cv2


class crop_values:
    x_start = 250
    x_end = 400
    y_start = 100
    y_end = 300

def pre_process_image(img, window_height, window_width):

    # gray = cv2.medianBlur(gray,3)
    # cv2.imshow("grey",gray)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # edges = cv2.Canny(img, 100, 200)

    # result is dilated for marking the corners, not important
    # dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # filename = "{}.png".format(os.getpid())
    # cv2.imwrite(filename, gray)

    img_height, img_width, img_colors = img.shape
    scale_w = float(window_width) / float(img_width)
    scale_h = float(window_height) / float(img_height)
    scale = min([scale_w, scale_h])

    if scale == 0:
        scale = 1

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def process_image_for_ocr(img):
    # img_cropped = cv2.rectangle(img, (crop_x_start, crop_y_start), (crop_x_end, crop_y_end), (255,0,0), 1)

    gray = img[crop_values.y_start:crop_values.y_end, crop_values.x_start: crop_values.x_end]

    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = cv2.threshold(gray, 100, 255,
                         cv2.THRESH_TOZERO)[1]

    return gray
