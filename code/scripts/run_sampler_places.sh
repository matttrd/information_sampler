python hyperoptim.py -c "python main_CVPR.py with save=True j=20 dataset.name='places_lt' exp='CVPR_sampler' tfl=True" -p '{ "sampler":["invtunnel"], "temperature":[0.1,1.0,10.0], "arch":["resnet18"], "adjust_classes":["True"], "normalizer":["False"]}' --gpus '[1,2]' -j 2 -r
#python hyperoptim.py -c "python main_CVPR.py with save=True j=20 dataset.name='places_lt' exp='CVPR_sampler' tfl=True" -p '{ "sampler":["default","class"], "arch":["resnet18"]}' --gpus '[0]' -j 1 -r