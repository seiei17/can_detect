import torch
import random
import numpy as np

from utils.util_custom import load_model, select_device, letterbox, non_max_suppression, scale_coords


class Inference:
    def __init__(self,
                 weights='./weight/best.pt',
                 imgsz=640,
                 devices='0',
                 conf_thres=.4,
                 iou_thres=.5):
        self.device = select_device(devices)
        self.model = load_model(filepath=weights, device=self.device)
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def inference(self, img0):

        res = list()

        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init image

        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = torch.div(img, 255.0)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)

        # Process detections
        for i, det in enumerate(pred):

            s = ''
            names = ['Big', 'Small']
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                # Rescale boxes from img_size to img0 size
                det[:, : 4] = scale_coords(img.shape[2:], det[:, : 4], img0.shape).round()

                # Print results
                for c in det[:, -1].detach().unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    res.append([int(cls), int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                    # if True:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     with open('./result/result' + '.txt', 'a') as f:
                    #         f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    #
                    # if True:
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    #     plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=3)
        return res


if __name__ == '__main__':
    import cv2
    import os

    weights = './weight/best.pt'
    inference = Inference(weights=weights,
                          devices='cpu')


    prefix = './images/'
    pics = os.listdir('./images')
    for pic in pics:
        pic_path = prefix + pic
        img0 = cv2.imread(pic_path)
        res = inference.inference(img0)
        print(res)
        break
