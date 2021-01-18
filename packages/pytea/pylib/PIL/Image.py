import LibCall
import random


class Image:
    def __init__(self):
        self._channel = 1
        self.size = (0, 0)
        self.height = 0
        self.width = 0
        self.mode = "L"

    def _setSize(self, channel, width, height):
        self._channel = channel
        self.width = width
        self.height = height
        self.size = (width, hight)
        LibCall.builtins.setSize(self, (channel, width, height))

    def copy(self):
        im = Image()
        im._setSize(self._channel, self.width, self.height)
        im.mode = self.mode
        return im

    def convert(self, mode=None, *args, **kwargs):
        if mode is None:
            return self
        elif len(mode) == 1:
            im = Image()
            im._setSize(1, self.width, self.height)
            im.mode = mode
            return im
        elif mode == "RGBA" or mode == "CMYK":
            im = Image()
            im._setSize(4, self.width, self.height)
            im.mode = mode
            return im
        else:
            im = Image()
            im._setSize(3, self.width, self.height)
            im.mode = mode
            return im

    def transform(self, size, method, data=None, resample=0, fill=1, fillcolor=None):
        if (
            method is not EXTENT
            and method is not AFFINE
            and method is not PERSPECTIVE
            and method is not QUAD
            and method is not MESH
            and not isinstance(method, ImageTransformHandler)
            and not hasattr(method, "getdata")
        ):
            raise Exception("unknown method type")

        if len(self.mode) == 1:
            im = Image()
            im._setSize(1, size[0], size[1])
            im.mode = self.mode
            return im
        elif self.mode == "RGBA" or mode == "CMYK":
            im = Image()
            im._setSize(4, size[0], size[1])
            im.mode = self.mode
            return im
        else:
            im = Image()
            im._setSize(3, size[0], size[1])
            im.mode = self.mode
            return im


def new(mode, size, color=0):
    if len(mode) == 1:
        im = Image()
        im._setSize(1, size[0], size[1])
        im.mode = mode
        return im
    elif mode == "RGBA" or mode == "CMYK":
        im = Image()
        im._setSize(4, size[0], size[1])
        im.mode = mode
        return im
    else:
        im = Image()
        im._setSize(3, size[0], size[1])
        im.mode = mode
        return im


class ImageTransformHandler:
    pass


def open(fp, mode="r"):
    im = Image()
    # make symbolic image
    im._setSize(
        random.randint(1, 4), random.randint(1, 10000), random.randint(1, 10000)
    )
    return im


def blend(im1, im2, alpha):
    LibCall.PIL.blend(im1, im2, alpha)  # just adds constraints, doesn't return obj.
    im = im1.copy()
    return im


NEAREST = 0
NONE = 0
BOX = 4
BILINEAR = 2
LINEAR = 2
HAMMING = 5
BICUBIC = 3
CUBIC = 3
LANCZOS = 1
ANTIALIAS = 1

# transforms
AFFINE = 0
EXTENT = 1
PERSPECTIVE = 2
QUAD = 3
MESH = 4
