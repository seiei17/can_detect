import cv2

from inference import Inference

if __name__ == '__main__':
    inf = Inference(weights='./weight/best.pt',
                    devices='cpu')
    img0 = cv2.imread('images/000002.jpg')
    res = inf.inference(img0)
    print(res)

    for obj in res:
        if obj[0] == 0:
            cv2.rectangle(img0, (obj[1], obj[2]), (obj[3], obj[4]), color=(0, 255, 0))
        if obj[0] == 1:
            cv2.rectangle(img0, (obj[1], obj[2]), (obj[3], obj[4]), color=(0, 0, 255))

    cv2.imshow('img', img0)
    if cv2.waitKey(0) == ord('q'):
        exit(0)
