from functools import wraps
from exptutils import *
# import tensorflow as tf
import numpy as np
import scipy.misc 
from exptutils import get_num_classes
import torch
from torch.utils.data import Subset

class Hook():
    "Base class for hooks (loggers for now)"

    def on_train_begin(self, **kwargs):
        "To initialize the callbacks."
        pass
    def on_epoch_begin(self, **kwargs):
        "At the beginning of each epoch."
        pass
    def on_batch_begin(self, **kwargs):
        "Called at the beginning of the batch."
        pass
    def on_batch_end(self, **kwargs):
        "Called at the end of the batch."
        pass
    def on_epoch_end(self, **kwargs):
        "Called at the end of an epoch."
        return False
    def on_train_end(self, **kwargs):
        "Useful for cleaning up things and saving files/models."
        pass


# register hooks
def register_hooks(ctx, other_hooks=[]):
    opt = ctx.opt
    hooks = []
    if opt['fl']:
        hooks.append(fLogger())
    if opt['tfl']:
        hooks.append(tfLogger())
    if opt['dbl']:
        hooks.append(dbLogger())

    ctx.hooks = hooks + other_hooks

def batch_hook(ctx, mode):
    def _batch_hook(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # pre-hook
            for hook in ctx.hooks:
                hook.on_batch_begin(ctx, mode)

            out = func(*args, **kwargs)

            # post-hook
            for hook in ctx.hooks:
                hook.on_batch_end(ctx, out, mode)
            return out
        return wrapped
    return _batch_hook


def epoch_hook(ctx, mode):
    def _epoch_hook(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # pre-hook
            for hook in ctx.hooks:
                hook.on_epoch_begin(ctx, mode)

            out = func(*args, **kwargs)

            # post-hook
            for hook in ctx.hooks:
                hook.on_epoch_end(ctx, out, mode)
            return out
        return wrapped
    return _epoch_hook    

def train_hook(ctx):
    def _train_hook(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # pre-hook
            for hook in ctx.hooks:
                hook.on_train_begin(ctx)

            out = func(*args, **kwargs)

            # post-hook
            for hook in ctx.hooks:
                hook.on_train_end(ctx)
            return out
        return wrapped
    return _train_hook


class tfLogger(Hook):

    def get_most_osc(self, ctx, k):
        return torch.topk(ctx.cum_sum_diff, k)

    def get_topk(self, ctx, k, largest=True):
        return torch.topk(ctx.cum_sum, k, largest=largest)

    def get_images(self, ctx, indices):
        if 'imagenet' not in ctx.opt['dataset']:
            dt = ctx.train_loader.dataset
            if isinstance(dt, Subset):
                images =  dt.dataset.data.data[indices.squeeze().cpu().numpy()]
            else:
                images =  dt.data.data[indices.squeeze().cpu().numpy()]
        else: # ImageFolder
            dt = ctx.train_loader.dataset
            if isinstance(dt, Subset):
                dt = dt.dataset.data
            else:
                dt = dt.data
            idx = indices.squeeze().cpu().numpy().tolist()
            tmp = [dt.samples[i] for i in idx]
            paths = list(map(lambda x: x[0], tmp))
            images = [np.array(dt.loader(p)) for p in paths]
            images = np.stack(images)
            images = images.transpose(0,3,1,2)
        return images

    def on_train_begin(self, ctx):
        global tf
        import tensorflow as tf
        from pathlib import Path
        self.home = str(Path.home())
        self.logger = dict()
        #logdir = os.path.join(self.home + '/tflogs', ctx.opt['arch'])
        logdir = os.path.join(self.home + '/tflogs', ctx.opt['filename'])
        self.logger['train'] = _tfLogger(os.path.join(logdir, 'train'))
        self.logger['train_clean'] = _tfLogger(os.path.join(logdir, 'train_clean'))
        self.logger['val'] = _tfLogger(os.path.join(logdir, 'val'))
        self.steps = 0
        ctx.ex.info.setdefault("tensorflow", {}).setdefault(
                "logdirs", []).append(logdir)
        # sess = tf.InteractiveSession()
        # sess.run(tf.global_variable_initializers())
        # ctx.sess = sess
        # ctx.summaries = []

    def on_epoch_begin(self, ctx, mode):
        pass

    def on_train_end(self, ctx):
        pass

    def on_batch_begin(self, ctx, mode):
        pass

    def on_batch_end(self, ctx, out, mode):
    	pass
        # # 1. Log scalar values (scalar summary)
        # info = { 'loss': loss.item(), 'accuracy': accuracy.item() }

        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, step+1)

        # 2. Log values and gradients of the parameters (histogram summary)
        # if mode is not 'train' or mode is not 'val':
        #     self.logger[mode] = _tfLogger(os.path.join(self.home + '/tflogs', mode))

        # if mode is 'train':
        #     model = ctx.model
        #     if ctx.i % ctx.opt['print_freq'] == 0:
        #         for tag, value in model.named_parameters():
        #             tag = tag.replace('.', '/')
        #             self.logger[mode].histo_summary(tag, value.data.view(-1,1).cpu().numpy(), self.steps)
        #             self.logger[mode].histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), self.steps)
        #     self.steps += 1
            # # 3. Log training images (image summary)
            # info = { 'osc_images':ctx.images['osc'][:50].cpu().numpy() }

            # for tag, images in info.items():
            #     self.logger.image_summary(tag, images, ctx.i+1)

            # info = { 'difficult_images':ctx.images['diff'][:50].cpu().numpy() }

            # for tag, images in info.items():
            #     self.logger.image_summary(tag, images, ctx.i+1)

    def on_epoch_end(self, ctx, out, mode):
        # prefix = mode + '.'
        # for k,v in out.items():
        #     self.logger[mode].scalar_summary(prefix + k, v, ctx.epoch)
        for k,v in out.items():
            # if k not in ctx.summaries:
            #     ctx.summaries.append(k)
            #     tf.summary.scalar(k,v)
            #     merged_summary_op = tf.summary.merge_all()
            #     ctx.summary = sess.run(merged_summary_op)
            self.logger[mode].scalar_summary(k, v, ctx.epoch)



        # if mode is 'train':

        # info = { 'images':ctx.images}
        # for tag, images in info.items():
        #     self.logger[mode].image_summary(tag, images, ctx.i+1)

        #     num_classes = get_num_classes(ctx.opt)
        #     self.logger[mode].histo_summary('weights_hist', None, ctx.epoch, numpy_hist=ctx.histograms['total'][0])
        #     for cl in range(num_classes):
        #         self.logger[mode].histo_summary('weights_hist_cl_' + str(cl), None, ctx.epoch, numpy_hist=ctx.histograms[str(cl)][0])

        #     K = 10
        #     # oscillating samples
        #     _, indices = self.get_most_osc(ctx, K)

        #     images = self.get_images(ctx, indices)
        #     info = { 'osc_images':images }

        #     for tag, images in info.items():
        #         self.logger[mode].image_summary(tag, images, ctx.epoch)

        #     # difficult samples
        #     _, indices = self.get_topk(ctx, K)
        #     images = self.get_images(ctx, indices)
        #     info = { 'difficult_images':images}

        #     for tag, images in info.items():
        #         self.logger[mode].image_summary(tag, images, ctx.epoch)

        #     # easy samples
        #     _, indices = self.get_topk(ctx, K, largest=False)

        #     images = self.get_images(ctx, indices)
        #     info = { 'easy_images':images}

        #     for tag, images in info.items():
        #         self.logger[mode].image_summary(tag, images, ctx.epoch)


class dbLogger(Hook):
    def on_train_begin(self, ctx):
        pass

    def on_batch_begin(self, ctx, mode):
        pass

    def on_batch_end(self, ctx, out, mode):
        pass

    def on_epoch_begin(self, ctx, mode):
        pass

    def on_epoch_end(self, ctx, out, mode):
        # ex = ctx.ex
        # metrics = ctx.metrics
        prefix = mode + '.'
        for k,v in out.items():
            ctx.ex.log_scalar(prefix + k, v)

    def on_train_end(self, ctx):
        pass

class fLogger(Hook):
    def on_train_begin(self, ctx):
        logger = create_logger(ctx)
        ctx.ex.logger = logger

    def on_batch_begin(self, ctx, mode):
        pass

    def on_batch_end(self, ctx, out, mode):
        if ctx.i % ctx.opt['print_freq'] == 0:
            if mode is 'train':
                ss = dict(e=ctx.epoch, i=0, train=True)
            elif mode == 'train_clean':
                ss = dict(e=ctx.epoch, i=0, train_clean=True)
            else:
                ss = dict(e=ctx.epoch, i=0, val=True)
            if len(out) > 1 and not isinstance(out, dict):
                out = out[0]
            ss.update(**out)
            ctx.ex.logger.info('[LOG] ' + json.dumps(ss))

    def on_epoch_begin(self, ctx, mode):
        pass

    def on_epoch_end(self, ctx, out, mode):
        if mode is 'train':
            ss = dict(e=ctx.epoch, i=0, train=True)
        elif mode == 'train_clean':
            ss = dict(e=ctx.epoch, i=0, train_clean=True)
        else:
            ss = dict(e=ctx.epoch, i=0, val=True)

        ss.update(**out)
        ctx.ex.logger.info('[SUMMARY] ' + json.dumps(ss))

    def on_train_end(self, ctx):
        pass


try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class _tfLogger(object):
    def __init__(self, log_dir='./tflogs'):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, numpy_hist=None, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        if numpy_hist is None:
            counts, bin_edges = np.histogram(values, bins=bins)
        else:
            counts, bin_edges = numpy_hist[0], numpy_hist[1]

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(bin_edges[0])
        hist.max = float(bin_edges[-1])
        hist.num = int(np.sum(counts))
        #hist.sum = float(np.sum(values))
        #hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush() 