import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def make_folder(path, version):
    if not os.path.exists(os.path.join(path, version)):
        os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return "".join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 19:  # CelebAMask-HQ
        cmap = np.array(
            [
                (0, 0, 0),
                (204, 0, 0),
                (76, 153, 0),
                (204, 204, 0),
                (51, 51, 255),
                (204, 0, 204),
                (0, 255, 255),
                (51, 255, 255),
                (102, 51, 0),
                (255, 0, 0),
                (102, 204, 0),
                (255, 255, 0),
                (0, 0, 153),
                (0, 0, 204),
                (255, 51, 153),
                (0, 204, 204),
                (0, 51, 0),
                (255, 153, 51),
                (0, 204, 0),
            ],
            dtype=np.uint8,
        )
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=19):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def tensor2label(label_tensor, n_label, imtype=np.uint8):
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    # label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    label_numpy = label_tensor.numpy()
    label_numpy = label_numpy / 255.0

    return label_numpy


def generate_label(inputs, imsize, n_classes):
    pred_batch = []
    for input in inputs:
        input = input.view(1, n_classes, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        p = p.view(1, imsize, imsize)
        label_batch.append(tensor2label(p, n_classes))

    label_batch = np.array(label_batch)
    label_batch = torch.from_numpy(label_batch)

    return label_batch


def generate_label_plain(inputs, imsize, n_classes):
    pred_batch = []
    for input in inputs:
        input = input.view(1, n_classes, imsize, imsize)
        pred = np.squeeze(input.data.max(1)[1].cpu().numpy(), axis=0)
        # pred = pred.reshape((1, 512, 512))
        pred_batch.append(pred)

    pred_batch = np.array(pred_batch)
    pred_batch = torch.from_numpy(pred_batch)

    label_batch = []
    for p in pred_batch:
        label_batch.append(p.numpy())

    label_batch = np.array(label_batch)

    return label_batch


def cross_entropy2d(input, target, weight=None, reduction="mean"):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, reduction=reduction, ignore_index=250
    )
    return loss


class PriorityQueue(deque):

    def __init__(self, iterable=[], maxlen=None):
        super().__init__(iterable, maxlen)

    def add(self, priority, obj):
        if len(self) == 0:
            self.append((priority, obj))
            return True
        for i, (p, o) in enumerate(self):
            if priority < p:
                if len(self) == self.maxlen:
                    self.pop()
                self.insert(i, (priority, obj))
                return True
        if self.maxlen is None or len(self) < self.maxlen:
            self.append((priority, obj))
        return False


class SaveBestModel:

    def __init__(self, model_save_path, maxlen=None):
        self.maxlen = maxlen
        self.model_save_path = model_save_path
        self.model_priority_queue = PriorityQueue(maxlen=maxlen)
        self.last_queue_update = 0
        self.best_val = float("inf")
        self.last_saved = []

    def put(self, priority, step, state_dict):
        self.model_priority_queue.add(priority, (step, state_dict))
        if priority < self.best_val:
            self.last_queue_update = 0
            self.best_val = priority
        else:
            self.last_queue_update += 1

    def save(self):
        old_models = os.listdir(self.model_save_path)
        to_save = []
        to_rename = []
        to_remove = [self.filter_step_from_fname(fn) for fn in old_models]

        for i, (score, (step, state_dict)) in enumerate(self.model_priority_queue):
            ranking = f"top_{str(i).zfill(2)}"
            fname = f"{ranking}_score_{score:.2f}_step_{str(step).zfill(7)}.pth"
            filter_fname = self.filter_step_from_fname(fname)
            if filter_fname in to_remove:
                to_remove.remove(filter_fname)
                old_fname = [fn for fn in old_models if filter_fname in fn][0]
                old_ranking = old_fname[: old_fname.index("_score_")]
                if old_ranking != ranking:
                    new_fname = old_fname.replace(old_ranking, ranking)
                    to_rename.append((old_fname, new_fname))
            else:
                to_save.append((fname, state_dict))

        # remove old files
        for r in to_remove:
            fname = [fn for fn in old_models if r in fn][0]
            os.remove(os.path.join(self.model_save_path, fname))

        # Rename if score changed
        for old_fname, new_fname in to_rename:
            os.rename(
                os.path.join(self.model_save_path, old_fname),
                os.path.join(self.model_save_path, new_fname),
            )

        # save new files
        for fname, state_dict in to_save:
            torch.save(state_dict, os.path.join(self.model_save_path, fname))

    @classmethod
    def filter_step_from_fname(cls, fname):
        start_idx = fname.index("_score_")
        end_idx = fname.index(".pth")
        return fname[start_idx:end_idx]
