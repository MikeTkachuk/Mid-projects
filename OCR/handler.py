import sys
import numpy as np
import torch
from torchvision import transforms

sys.path.append(r'C:\Users\Mykhailo_Tkachuk\PycharmProjects\Mid-projects\Cloned')
from serve.ts.torch_handler.vision_handler import VisionHandler


class Handler(VisionHandler):
    def __init__(self):
        super().__init__()
        self.image_processing = lambda x: torch.tensor(np.asarray(x))

    def postprocess(self, data):
        def to_str(s):
            return ''.join([chr(i) for i in s])
        return [to_str(data)]