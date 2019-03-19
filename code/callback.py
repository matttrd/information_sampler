class Callback():
    "Base class for callbacks (loggers for now)"

    def on_train_begin(self, **kwargs:Any)->None:
        "To initialize the callbacks."
        pass
    def on_epoch_begin(self, **kwargs:Any)->None:
        "At the beginning of each epoch."
        pass
    def on_batch_begin(self, **kwargs:Any)->None:
    	"Called at the beginning of the batch."
        pass
    def on_batch_end(self, **kwargs:Any)->None:
        "Called at the end of the batch."
        pass
    def on_epoch_end(self, **kwargs:Any)->bool:
        "Called at the end of an epoch."
        return False
    def on_train_end(self, **kwargs:Any)->None:
        "Useful for cleaning up things and saving files/models."
        pass
    
    # def get_state(self, minimal:bool=True):
    #     "Return the inner state of the `Callback`, `minimal` or not."
    #     to_remove = ['exclude', 'not_min'] + getattr(self, 'exclude', []).copy()
    #     if minimal: to_remove += getattr(self, 'not_min', []).copy()
    #     return {k:v for k,v in self.__dict__.items() if k not in to_remove}
    
    # def  __repr__(self): 
    #     attrs = func_args(self.__init__)
    #     to_remove = getattr(self, 'exclude', [])
    #     list_repr = [self.__class__.__name__] + [f'{k}: {getattr(self, k)}' for k in attrs if k != 'self' and k not in to_remove]
    #     return '\n'.join(list_repr) 