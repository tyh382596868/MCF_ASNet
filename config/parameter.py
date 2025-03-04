import argparse
import yaml
import importlib

class Parameter:
    def __init__(self, config_path):
        with open(config_path, 'rb') as file:
            self.config = yaml.safe_load(file)

    def __getattr__(self, item):
        return self.config.get(item)

def import_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

# if __name__=='__main__':
#     print('argparse')
#     parse = argparse.ArgumentParser(description='training script')
#     # print(parse)

#     parse.add_argument('--file',type=str,default='/mnt/data/optimal/tangyuhang/workspace/iopen/ai4optical/phynet_git/phynet/config/config.yaml')

#     opt = parse.parse_args()
#     print(opt.file)

#     with open(opt.file, 'r') as file:
#         config = yaml.safe_load(file) 

#     print(config.get('device')) 