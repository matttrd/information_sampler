import torch
import numpy as np
import foolbox
import foolbox.attacks.RandomStartProjectedGradientDescentAttack as PGD
import torchnet as tnt 

p = argparse.ArgumentParser('unadversarial training',
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument('-i', '--input', type=str, default='', help='location of saved models')
p.add_argument('--dataset', type=str, default='cifar10', help='dataset')
p.add_argument('-g', type=int, default=0, help='cuda device')
p.add_argument('-b', type=int, default=128, help='batch-size')
p.add_argument('-s', type=int, default=42, help='seed')

opt = vars(p.parse_known_args()[0])


np.random.seed(opt['s'])
torch.manual_seed(opt['s'])
torch.cuda.manual_seed_all(opt['s'])


def main():
	d = th.load(f, map_location=lambda storage, loc: storage.cuda())
	model = getattr(models, d['arch'])({'d': 0.}).cuda(opt['g'])
	model.load_state_dict(d['state_dict'])
	fmodel = foolbox.models.PyTorchModel(model, bounds=(0, 1), num_classes=models.get_num_classes(opt))
	attack = PGD(fmodel)
	train_loader, val_loader = load_data(name=opt['dataset'], opt=opt)
	adv_conf_mat = tnt.meter.ConfusionMeter(models.get_num_classes(opt), normalized=True)
	errors = ClassErrorMeter(topk=[1])

	for i, (x,y) in enumerate(val_loader):
		x = x.cuda()
		y = y.cuda()
		adversarial = attack(x, y)
		pred = np.argmax(fmodel.predictions(adversarial))
		adv_conf_mat.add(pred, y.data.cpu().numpy())
		errors.add(pred, y.data.cpu().numpy())
	conf_mat = adv_conf_mat.value()
	top1 = errors.value()[0]

	print('Error: ', top1)
	# print('adversarial class', np.argmax(fmodel.predictions(adversarial)))
	