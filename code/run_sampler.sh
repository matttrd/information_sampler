#python hyperoptim.py -c "python main_epoch.py with save=True wd=5e-4 dataset.name='imagenet_lt' exp='sampler_epoch' tfl=True use_train_clean=True o='/home/matteo/information_sampler/results/' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90" -p '{ "sampler":["alternate","tunnel","invtunnel"], "dataset.name":["imagenet_lt"], "arch":["resnet18","resnet34"], "adjust_classes":["True"]}' --gpus '[1,2]' -r -j 2
#python hyperoptim.py -c "python main_lt.py with save=True wd=5e-4 j=12 dataset.name='imagenet_lt' exp='sampler_epoch' tfl=True use_train_clean=True o='/home/matteo/information_sampler/results/' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90" -p '{ "sampler":["tunnel","invtunnel"], "dataset.name":["imagenet_lt"], "temperature":[0.1,1.0], "arch":["resnet18","resnet34"], "adjust_classes":["False","True"]}' --gpus '[1,2]' -j 2 -r
python hyperoptim.py -c "python main_lt.py with b=64 save=True wd=5e-4 j=12 dataset.name='imagenet_lt' exp='sampler_epoch' tfl=True o='/home/matteo/information_sampler/results/' lrs='[[0,0.05],[30,0.005],[60,0.0005],[80,0.00005]]' epochs=90" -p '{ "sampler":["invtunnel"], "temperature":[0.1], "arch":["resnet101"], "adjust_classes":["False"]}' --gpus '[1]' -j 2 -r