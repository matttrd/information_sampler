python main_lt.py with b=256 wd=5e-4 arch='resnet10' marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. dataset.name='imagenet_lt' save=True tfl=True
python main_lt.py with b=256 wd=5e-4 arch='resnet10' marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. dataset.name='imagenet_lt' save=True tfl=True adjust_classes=True
python main_lt.py with b=256 wd=5e-4 arch='resnet10' marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. dataset.name='imagenet_lt' save=True tfl=True bce=True
python main_lt.py with b=256 wd=5e-4 arch='resnet10' marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. dataset.name='imagenet_lt' save=True tfl=True adjust_classes=True bce=True
#python main_lt.py with g=2 wd=5e-4 marker='tunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='tunnel' temperature=10. save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=10. save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='plain_att' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 modatt=True save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='plain' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=5. save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=2. save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=1. save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=0.01 save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=0.1 save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperature=10. modatt=True save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' temperatures='[50,0.1]' save=True tfl=True
# python main_lt.py with g=2 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 sampler='invtunnel' dyncount=True save=True tfl=True
#python main_lt.py with g=2 b=256 wd=5e-4 marker='plain' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 save=True tfl=True
#python main_lt.py with g=2 b=256 wd=5e-4 marker='plain' lrs='[[0,0.1],[10,0.01],[20,0.001]]' epochs=30 j=10 nesterov=False
# python main_lt.py with b=256 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=10. save=True tfl=True
# python main_lt.py with b=256 wd=5e-4 marker='invtunnel' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. save=True tfl=True
#python main_lt_adv.py with b=256 wd=5e-4 k=7 eps=3.5 marker='invtunnel_adj_x10_adv' lrs='[[0,0.1],[30,0.01],[60,0.001],[80,0.0001]]' epochs=90 j=10 sampler='invtunnel' temperature=1. adjust_classes=True save=True tfl=True