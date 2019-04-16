import torch
import numpy as np
import foolbox
from foolbox.attacks import RandomStartProjectedGradientDescentAttack as PGD
import torchnet as tnt 
import argparse
import models
from light_loader import data_ingredient, load_data
from sacred import Experiment
from torchnet.meter import ClassErrorMeter, ConfusionMeter

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

@ex.automain
def main():
    opt=init()
    np.random.seed(opt['s'])
    torch.manual_seed(opt['s'])
    torch.cuda.manual_seed_all(opt['s'])

    print('Loading model...')
    d = torch.load(opt['input'], map_location=lambda storage, loc: storage.cuda())
    opt['d'] = 0.
    model = getattr(models, d['arch'])(opt).cuda(opt['g'])
    model.load_state_dict(d['state_dict'])
    model.eval()
    fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=models.get_num_classes(opt))
    attack = PGD(fmodel, distance=foolbox.distances.Linfinity)
    train_loader, val_loader, _ = load_data(name=opt['dataset'], opt=opt)
    adv_conf_mat = ConfusionMeter(models.get_num_classes(opt), normalized=True)
    errors = ClassErrorMeter(topk=[1])

    for i, (x,y) in enumerate(val_loader):
        adversarial = attack(x.cpu().numpy()[0], y.cpu().numpy()[0], epsilon=6./255., binary_search=False)
        pred = np.argmax(fmodel.predictions(adversarial))
        pred = torch.from_numpy(pred[np.newaxis, ...]).long()
        adv_conf_mat.add(pred.data, y.data)
        #errors.add(pred.data, y.data)
    conf_mat = adv_conf_mat.value()
    #top1 = errors.value()[0]
    print('Error: ', 1-np.trace(conf_mat) / np.sum(conf_mat))
    # print('adversarial class', np.argmax(fmodel.predictions(adversarial)))
    