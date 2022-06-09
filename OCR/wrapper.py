import torch
import torch.nn.functional as F
import numpy as np
from math import *
import sys
import cv2

sys.path.append(r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Mid-projects\Cloned')
sys.path.append(r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Mid-projects\Cloned\ocr_pytorch')
from detect.ctpn_model import CTPN_Model
from detect import config
from detect.ctpn_utils import resize
from detect.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox, nms, TextProposalConnectorOriented

from recognize.crnn_recognizer import PytorchOcr
from recognize.crnn import CRNN
from recognize import config as config_r


class ModelWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ctpn = CTPN_Model()
        self.recognizer = PytorchOcr(None)
        self.crnn = CRNN(config_r.imgH, 1, self.recognizer.nclass, 256)
        self.recognizer.model = self.crnn
        if not torch.cuda.is_available():
            gpu = False
        self.device = torch.device('cuda:0' if gpu else 'cpu')

    def _get_det_boxes(self, image, display=True, expand=True, height=720, prob_thresh=0.5):
        image = resize(image, height=height)
        image_r = image.copy()
        image_c = image.copy()
        h, w = image.shape[:2]
        image = image.astype(np.float32) - config.IMAGE_MEAN
        image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            image = image.to(self.device)
            cls, regr = self.ctpn(image)
            cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
            regr = regr.cpu().numpy()
            anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
            bbox = bbox_transfor_inv(anchor, regr)
            bbox = clip_box(bbox, [h, w])
            # print(bbox.shape)

            fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
            # print(np.max(cls_prob[0, :, 1]))
            select_anchor = bbox[fg, :]
            select_score = cls_prob[0, fg, 1]
            select_anchor = select_anchor.astype(np.int32)
            # print(select_anchor.shape)
            keep_index = filter_bbox(select_anchor, 16)

            # nms
            select_anchor = select_anchor[keep_index]
            select_score = select_score[keep_index]
            select_score = np.reshape(select_score, (select_score.shape[0], 1))
            nmsbox = np.hstack((select_anchor, select_score))
            keep = nms(nmsbox, 0.3)
            # print(keep)
            select_anchor = select_anchor[keep]
            select_score = select_score[keep]

            # text line-
            textConn = TextProposalConnectorOriented()
            text = textConn.get_text_lines(select_anchor, select_score, [h, w])

            # expand text
            if expand:
                for idx in range(len(text)):
                    text[idx][0] = max(text[idx][0] - 10, 0)
                    text[idx][2] = min(text[idx][2] + 10, w - 1)
                    text[idx][4] = max(text[idx][4] - 10, 0)
                    text[idx][6] = min(text[idx][6] + 10, w - 1)

            # print(text)
            if display:
                blank = np.zeros(image_c.shape, dtype=np.uint8)
                for box in select_anchor:
                    pt1 = (box[0], box[1])
                    pt2 = (box[2], box[3])
                    blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1)
                image_c = image_c + blank
                image_c[image_c > 255] = 255
                for i in text:
                    s = str(round(i[-1] * 100, 2)) + '%'
                    i = [int(j) for j in i]
                    cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                    cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                    cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                    cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                    cv2.putText(image_c, s, (i[0] + 13, i[1] + 13),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 0, 0),
                                2,
                                cv2.LINE_AA)
                # dis(image_c)
            # print(text)
        return text, image_c, image_r

    @staticmethod
    def sort_box(box):
        """
        对box进行排序
        """
        box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        return box

    @staticmethod
    def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
        height, width = img.shape[:2]
        heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
        widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
        matRotation[0, 2] += (widthNew - width) // 2
        matRotation[1, 2] += (heightNew - height) // 2
        imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
        pt1 = list(pt1)
        pt3 = list(pt3)

        [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
        [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
        ydim, xdim = imgRotation.shape[:2]
        imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
                 max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

        return imgOut

    def _charRec(self, img, text_recs, adjust=False):
        """
        加载OCR模型，进行字符识别
        """
        results = {}
        xDim, yDim = img.shape[1], img.shape[0]

        for index, rec in enumerate(text_recs):
            xlength = int((rec[6] - rec[0]) * 0.1)
            ylength = int((rec[7] - rec[1]) * 0.2)
            if adjust:
                pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
                pt4 = (rec[4], rec[5])
            else:
                pt1 = (max(1, rec[0]), max(1, rec[1]))
                pt2 = (rec[2], rec[3])
                pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
                pt4 = (rec[4], rec[5])

            degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

            partImg = self.dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
            # dis(partImg)
            if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
                continue
            text = self.recognizer.recognize(partImg)
            if len(text) > 0:
                results[index] = [rec]
                results[index].append(text)  # 识别文字

        return results

    def forward(self, x):
        def to_ansii(s):
            return [ord(s_) for s_ in s]

        text_recs, img_framed, image = self._get_det_boxes(np.asarray(x[0,...,:3]))
        text_recs = self.sort_box(text_recs)
        result = self._charRec(image, text_recs)
        text = [result[i][1] for i in sorted(list(result.keys()))]
        text = to_ansii(' '.join(text))

        return torch.tensor(text)
