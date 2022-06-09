from wrapper import ModelWrapper
import torch
import numpy as np
import cv2 as cv
from ocr import ocr

def to_ansii(s):
    return [ord(s_) for s_ in s]
def to_str(ans):
    return ''.join([chr(a) for a in ans])
print(torch.tensor(np.array(to_ansii('one_two2'))))

m = ModelWrapper()
m.ctpn.load_state_dict(torch.load('checkpoints/CTPN.pth', map_location=torch.device('cpu'))['model_state_dict'])
m.crnn.load_state_dict(torch.load('checkpoints/CRNN-1010.pth', map_location=torch.device('cpu')))
torch.save(m.state_dict(), 'serialized.pt')
m = ModelWrapper()
m.load_state_dict(torch.load('serialized.pt'))

im = cv.imread('in.png')
im = cv.cvtColor(im, cv.COLOR_BGRA2RGB)

print(m(im)[0])
print([i[1] for _, i in ocr(im)[0].items()])
cv.imwrite('out.png', m(im)[1],)