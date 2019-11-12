
SAVE_EPOCHS = [0,59,119,159]

TRAINING_DEFAULTS = {
    'cifar10': {
        "epochs": 180,
        #"b": 128,
        "save_epochs" : SAVE_EPOCHS,
        "wd":5e-4,
        #"lrs": '[[0,0.1],[60,0.02],[120,0.004],[160,0.0008]]'
    },
    'cifar100': {
        "epochs": 180,
       #"b": 128,
       "save_epochs" : SAVE_EPOCHS,
        "wd":5e-4,
        "lrs": '[[0,0.1],[60,0.02],[120,0.004],[160,0.0008]]'
    },
    'cinic': {
        "epochs": 150,
        #"b": 128,
        "save_epochs" : SAVE_EPOCHS,
        "wd":5e-4,
        "lrs": '[[0,0.1],[50,0.01],[100,0.001]]'
    },
    'imagenet': {
        "epochs": 350,
        #"b":256,
        "save_epochs" : SAVE_EPOCHS,
        "wd":1e-4,
        "lrs": '[[0,0.1],[150,0.01],[300,0.001]]'
    },
    'imagenet_lt': {
        #"epochs": 150,
        #"b": 256,
        "save_epochs" : SAVE_EPOCHS,
        "wd": 5e-4,
         #"lrs": '[[0,0.1],[50,0.01],[100,0.001]]'
    },
    'places_lt': {
        "epochs": 180,
        #"b": 256,
        "save_epochs" : SAVE_EPOCHS,
        "wd": 5e-4,
         "lrs": '[[0,0.1],[50,0.01],[100,0.001],[150,0.0001]]'
    }
}


def add_args_to_opt(dataset, opt):
    '''
    Set and OVERWRITES the default args
    '''
    defaults = TRAINING_DEFAULTS[dataset]
    for k,v in defaults.items():
        opt[k] = v
    return opt
