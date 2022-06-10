from wrapper import ModelWrapper
import torch
import os
import sys

handler = r"C:/Users/Mykhailo_Tkachuk/PycharmProjects/Mid-projects/Cloned/ocr_pytorch/handler.py"
model_file = 'wrapper.py'
model_name = 'modelwrapper'
serialized_file = 'serialized.pt'
model_store = r'C:/Users/Mykhailo_Tkachuk/PycharmProjects/Mid-projects/Cloned/ocr_pytorch/model_store'

m = ModelWrapper()
m.ctpn.load_state_dict(torch.load('checkpoints/CTPN.pth', map_location=torch.device('cpu'))['model_state_dict'])
m.crnn.load_state_dict(torch.load('checkpoints/CRNN-1010.pth', map_location=torch.device('cpu')))
torch.save(m.state_dict(), serialized_file)

command = 'torch-model-archiver --model-file {model_file} --model-name {model_name}'+ \
 ' --version 1.0 --serialized-file {serialized_file} --handler {handler}'+ \
 ' --export-path {model_store} -f'
run_server = 'torchserve --start --ncs --model-store {model_store} --models {model_name}.mar'

os.system(command.format(model_file=model_file, model_name=model_name, serialized_file=serialized_file,
                         handler=handler, model_store=model_store))
os.system(run_server.format(model_store=model_store, model_name=model_name))


r"""
Access the server via
 curl.exe http://127.0.0.1:8080/predictions/modelwrapper -T C:\Users\Mykhailo_Tkachuk\PycharmProjects\Mid-projects\Cloned\ocr_pyt
orch\in.png

"""