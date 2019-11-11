import torch
import numpy as np
import foolbox
from foolbox.attacks import RandomStartProjectedGradientDescentAttack as PGD
import torchnet as tnt 
import argparse
#import models
from light_loader import data_ingredient, load_data
from sacred import Experiment
from torchnet.meter import ClassErrorMeter, ConfusionMeter
from exptutils import *
from IPython import embed
ex = Experiment('attack', ingredients=[data_ingredient])

@ex.config
def cfg():
    input = ''
    g = 0
    b = 1
    s = 42
    j = 1

def init_opt():
    cfg = ex.current_run.config
    opt = dict()
    for k,v in cfg.items():
        opt[k] = v
    return opt

@data_ingredient.capture
def init(name):
    opt = init_opt()
    opt['dataset'] = name
    return opt

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self,x):
        out, _ = self.m(x)
        return out

@ex.automain
def main():
    opt=init()
    np.random.seed(opt['s'])
    torch.manual_seed(opt['s'])
    torch.cuda.manual_seed_all(opt['s'])

    print('Loading model...')
    d = torch.load(opt['input'], map_location=lambda storage, loc: storage.cuda())
    opt['d'] = 0.
    opt_m = d['opt']
    opt_m['dataset'] = opt['dataset']
    #model = getattr(models, d['arch'])(opt).cuda(opt['g'])
    model = create_and_load_model(opt_m)
    model.load_state_dict(d['state_dict'])
    model = ModelWrapper(model)
    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=get_num_classes(opt_m))
    attack = PGD(fmodel, distance=foolbox.distances.MeanSquaredDistance)
    train_loader, val_loader, _ = load_data(name=opt['dataset'], opt=opt)
    adv_conf_mat = ConfusionMeter(get_num_classes(opt_m), normalized=True)
    errors = ClassErrorMeter(topk=[1])

    for i, (x,y) in enumerate(val_loader):
        adversarial = attack(x.cpu().numpy()[0], y.cpu().numpy()[0], epsilon=1./255., binary_search=False, iterations=20)
        #adversarial = attack(x.cpu().numpy()[0], y.cpu().numpy()[0])
        if adversarial is None:
        	pred = y.cpu().numpy()[0]
        else:
        	pred = np.argmax(fmodel.predictions(adversarial))
        pred = torch.from_numpy(pred[np.newaxis, ...]).long()
        adv_conf_mat.add(pred.data, y.data)
        #errors.add(pred.data, y.data)
    conf_mat = adv_conf_mat.value()
    #top1 = errors.value()[0]
    print('Error: ', 1-np.trace(conf_mat) / np.sum(conf_mat))
    # print('adversarial class', np.argmax(fmodel.predictions(adversarial)))
    
