python hyperoptim.py -c "python main_CVPR.py with save=True j=4 dataset.name='cifar100' exp='CVPR_corr_labels' tfl=True" -p '{"sampler":["tunnel","invtunnel"],"b":[128], "temperature":[0.001,1.0,1000.0], "arch":["resnet18"],"normalizer":["True"],"corr_labels":[0.1,0.2,0.3]}' --gpus '[0,1,2]' -j 6 -r