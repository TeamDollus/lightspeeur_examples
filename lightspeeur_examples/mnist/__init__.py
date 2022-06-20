import random
import platform

import numpy as np

from tensorflow.keras import models, datasets, layers
from lightspeeur.layers import Conv2D, MaxPooling2D, ReLU
from lightspeeur.layers.quantization import quantize_image
from lightspeeur.models import ModelStageAdvisor, ModelConverter
from lightspeeur.drivers import Driver, Model


class Config:

    chip_id = '2803'
    batch_size = 256
    epochs = 1
    relu_calibration_steps = 32
    evaluation_steps = 10


cfg = Config()

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Lightspeeur models don't support single channel input. The channel size must be at least 3.
train_images = np.concatenate((train_images, train_images, train_images), axis=-1)
test_images = np.concatenate((test_images, test_images, test_images), axis=-1)

# Quantize the image to simulate chip-inputs
# NOTE: DO NOT quantize `test_images`. The quantization process will be automatically performed in the chip.
train_images = quantize_image(train_images)


def build_simple_conv_model(folded_shape=False):
    inputs = layers.Input(shape=(28, 28, 3))

    x = inputs
    for index, channel_size in enumerate([32, 64]):
        x = Conv2D(channel_size, (3, 3), cfg.chip_id,
                   bit_mask=2, quantize=folded_shape, use_bias=folded_shape,
                   name='conv{}/conv'.format(index))(x)
        if not folded_shape:
            x = layers.BatchNormalization(name='conv{}/batch_norm'.format(index))(x)
        x = ReLU(cfg.chip_id, quantize=folded_shape, name='conv{}/relu'.format(index))(x)
        x = MaxPooling2D(name='conv{}/pool'.format(index))(x)

    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='fc')(x)
    x = layers.Dense(10, activation='softmax', name='digit')(x)

    full_model = models.Model(inputs=inputs, outputs=x, name='simple_conv_model')
    return full_model


model = build_simple_conv_model()
folded_model = build_simple_conv_model(folded_shape=True)

print()
print(model.summary())
print()

compile_options = {
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy']
}
print('Preparing model stage advisor...')
advisor = ModelStageAdvisor(cfg.chip_id, model, compile_options,
                            folded_shape_model=folded_model,
                            cleanup_checkpoints=True)
while True:
    res = advisor.advance_stage()
    if res:
        advisor.propose(train_images,
                        train_labels,
                        epochs=cfg.epochs,
                        validation_split=0.2,
                        batch_size=cfg.batch_size,
                        relu_calibration_sample_steps=cfg.relu_calibration_steps,
                        fine_tune_folded_shape_model=True)
    else:
        break

print('Latest checkpoint is being saved...')
advisor.get_model().save('mnist-best.h5')

if platform.system() != 'Linux':
    raise NotImplementedError('Windows and macOS are not supported')

final_model = advisor.get_model()
final_model.load_weights('mnist-best.h5')

driver = Driver()
converter = ModelConverter(cfg.chip_id, final_model, graph={
    'simple_conv': [['conv0/conv', 'conv0/relu', 'conv0/pool'],
                    ['conv1/conv', 'conv1/relu', 'conv1/pool']]
}, debug=True)

model_path = converter.convert(driver)[0]

off_chip_model = models.Model(inputs=final_model.get_layer('flatten').input,
                              outputs=final_model.output,
                              name='off_chip_model')
print()
print(off_chip_model.summary())
print()

lightspeeur_model = Model(driver, cfg.chip_id, model_path)
with lightspeeur_model:
    for i in range(cfg.evaluation_steps):
        sample_index = random.randint(0, test_images.shape[0])

        sample_image = test_images[sample_index]
        sample_label = test_labels[sample_index]

        res = lightspeeur_model.evaluate(sample_image)
        predicted = off_chip_model(res)

        print()
        print('Actual: {}'.format(sample_label))
        print('Predicted: {}'.format(np.argmax(predicted[0])))
