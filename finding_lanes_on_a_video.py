import cv2
import numpy as np

# 5) TUR - Ortalamasını oluşturduğumuz takip çizgilerini oluşturmak için
#    koordinatlarını oluşturmamız gerekiyor.
#    ENG - We must get the coordinates of the averaged lines that to produce the detected lines.
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# 4) TUR - Şeritleri takip edecek çizgilerin ortalamasını alarak düzgün bir çizgi elde ediyoruz.
#    ENG - We take the average of detected lines to produce a smoother line.
def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

# 3) TUR - Görüntüyü tek kanala indirip grayscale olarak işlemek daha kolaydır.
#          Daha sonra GaussianBlur filtresi ve canny image ile görüntüdeki keskin kenarları tespit ediyoruz.
#    ENG - Since processing the image in one channel is computationally efficient, we use the grayscale.
#          After the GaussianBlur filter, we use the Canny() function to detect the edges.
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# 2) Şeritleri takip edecek çizgiyi çıkarıyoruz.
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

# 1)TUR - Görüntüde ilgilendiğimiz kısmı kesiyoruz.
#   ENG - We crop the part that we are interested in the image.
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)#
    return masked_image

# image = cv2.imread("test_image.jpg")
# lane_image = np.copy(image)
# canny_image = canny(lane_image)
# cropped_image = region_of_interest(canny_image)
# minLineLength=40
# maxLineGap=5
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength, maxLineGap)
# averaged_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("Detected Lane", combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("test1.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    minLineLength = 40
    maxLineGap = 5
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength, maxLineGap)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("Detected Lane", combo_image)
    if cv2.waitKey(5) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
