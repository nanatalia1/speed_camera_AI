import cv2
import imutils as imutils
import pytesseract as pytesseract
from ultralytics import YOLO
import os
from cascadeutils import generate_negative_description_file

path = r"C:\Users\mdaniele\PycharmProjects\ai_proj\Stanford_Car.v10-accurate-model_mergedallclasses-augmented_by3x.yolov8\test\images\002149_jpg.rf.810c92f66697e3a86c2f9f35dc1ae0ea.jpg"

# wczytanie
image = cv2.imread(path)
car_cascade = cv2.CascadeClassifier('cascade/cascade.xml')
model = YOLO("yolov8m.pt")


def processPlate(image=image):
    image = imutils.resize(image, width=500)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("greyed image", gray_image)
    cv2.waitKey(0)

    gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
    cv2.imshow("smoothened image", gray_image)
    cv2.waitKey(0)

    edged = cv2.Canny(gray_image, 30, 200)
    cv2.imshow("edged image", edged)
    cv2.waitKey(0)

    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image1 = image.copy()
    cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("contours", image1)
    cv2.waitKey(0)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    screenCnt = None
    image2 = image.copy()
    cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
    cv2.imshow("Top 30 contours", image2)
    cv2.waitKey(0)

    i = 7
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            screenCnt = approx
            x, y, w, h = cv2.boundingRect(c)
            new_img = image[y:y + h, x:x + w]
            cv2.imwrite('./' + str(i) + '.png', new_img)
            i += 1
            break
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("image with detected license plate", image)
    cv2.waitKey(0)

    Cropped_loc = './7.png'
    cv2.imshow("cropped", cv2.imread(Cropped_loc))
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    plate = pytesseract.image_to_string(Cropped_loc, lang='eng')
    print("Number plate is:", plate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def carsDetectionHAAR():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(cars) > 0:

        # Znajdź największy ROI
        max_area = 0
        max_roi = None

        for (x, y, w, h) in cars:
            area = w * h
            if area > max_area:
                max_area = area
                max_roi = (x, y, w, h)

        if max_roi is not None:
            # Pobierz współrzędne największego ROI
            x, y, w, h = max_roi

            # Wyciągnij ROI na podstawie współrzędnych ramki ograniczającej
            roi = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Wyświetl ROI
            cv2.imshow('ROI - Największy wykryty samochód', roi)
            cv2.imshow('Obraz z ROI', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            processPlate(roi)
        else:
            print("Nie można znaleźć ROI samochodu!")
    else:
        print("Nie wykryto samochodów na zdjęciu!")


def singleImageCarsDetectionYOLO():
    image = cv2.imread(path)
    results = model.predict(source=path, conf=0.5, save_crop=True, classes=2)
    for r in results:
        boxes = r.boxes.xywh
        if boxes.size(dim=0) == 0:
            print("Nie wykryto samochodu")
        for box in boxes:
            roi = image[round(box[1].item()):round(box[1].item()) + round(box[3].item()),
                  round(box[0].item()):round(box[0].item()) + round(box[2].item())]
            cv2.rectangle(image, (round(box[0].item()), round(box[1].item())), (
                round(box[1].item()) + round(box[3].item()), round(box[0].item()) + round(box[2].item())),
                          (0, 255, 0),
                          2)
            cv2.imshow("ROI", roi)
            cv2.imshow("Obraz", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            processPlate(roi)


singleImageCarsDetectionYOLO()
# carsDetectionHAAR()
# processPlate(image)
