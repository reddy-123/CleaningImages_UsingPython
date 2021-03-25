#!/usr/bin/env python
import argparse
import glob
import gzip
import logging
import numpy
import os.path
import PIL.Image
import sys

import clean
import train

LOGGER_NAME = 'backend.%s'
logger = logging.getLogger(LOGGER_NAME % __name__)
logger.addHandler(logging.StreamHandler(sys.stderr))

def load_test_images(limit=None, neighbors=5):
    image_specs = []
    xs = []
    for path in glob.glob('../../test/*.png')[:limit]:
        image_number = os.path.basename(path)[:-len('.png')]
        patches, shape = clean.x_from_image(path, neighbors)
        image_specs.append((image_number, shape))
        xs.append(patches)

    return image_specs, xs

def build_submission(model, limit, neighbors):
    imgs = []

    image_specs, xs = load_test_images(limit, neighbors)
    for (num, shape), patches in zip(image_specs, xs):
        predictions = model.predict(patches)
        shaped = clean.from_range(predictions.reshape(shape))
        img = PIL.Image.fromarray(shaped)
        imgs.append(img)

    return image_specs, imgs

def dump(image_spec, imgs, output_dir):
    with gzip.open(os.path.join(output_dir, 'submission.txt.gz'), 'w') as f:
        f.write('id,value\n')
        for (num, _), img in zip(image_spec, imgs):
            img.convert('L').save(os.path.join(output_dir, num + '.png'))

            pixels = numpy.array(img) / 255.0
            it = numpy.nditer(pixels, flags=['multi_index'])
            while not it.finished:
                pixel = it[0]
                i, j = it.multi_index
                f.write('{}_{}_{},{}\n'.format(num, i+1, j+1, pixel))
                it.iternext()



    model, params = clean.load_model(args.model)
    specs, imgs = build_submission(model, args.limit, params['neighbors'])
    dump(specs, imgs, args.output_dir)

    return 0

if __name__ == '__main__':
    sys.exit(main())
