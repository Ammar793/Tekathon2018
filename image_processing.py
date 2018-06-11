import cv2


class crop_values:
    x_start = 330
    x_end = 650
    y_start = 30
    y_end = 510

class crop_values_name:
    x_start = 330
    x_end = 650
    y_start = 420
    y_end = 510

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
    scale_w = float(window_width-10) / float(img_width)
    scale_h = float(window_height-10) / float(img_height)
    scale = min([scale_w, scale_h])

    if scale == 0:
        scale = 1

    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def process_image_for_ocr(img):
    gray = img[crop_values_name.y_start:crop_values_name.y_end, crop_values_name.x_start: crop_values_name.x_end]

    gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    gray = cv2.threshold(gray, 100, 255,
                         cv2.THRESH_TOZERO)[1]

    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return gray


def process_image_for_snack_detection(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    imag = img[crop_values.y_start:crop_values.y_end, crop_values.x_start: crop_values.x_end]
    #cv2.imshow("gray", imag)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return imag

def make_rectangle(img):
    img_cropped = cv2.rectangle(img, (crop_values.x_start, crop_values.y_start), (crop_values.x_end, crop_values.y_end),
                                (255, 0, 0), 1)
    return img_cropped