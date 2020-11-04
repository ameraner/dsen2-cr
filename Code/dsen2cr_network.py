import keras.backend as K
import tensorflow as tf
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add
from keras.models import Model, Input

K.set_image_data_format('channels_first')


def resBlock(input_l, feature_size, kernel_size, scale=0.1):
    """Definition of Residual Block to be repeated in body of network."""
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(input_l)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)

    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([input_l, tmp])


def DSen2CR_model(input_shape,
                  batch_per_gpu=2,
                  num_layers=32,
                  feature_size=256,
                  use_cloud_mask=True,
                  include_sar_input=True):
    """Definition of network structure. """

    global shape_n

    # define dimensions
    input_opt = Input(shape=input_shape[0])
    input_sar = Input(shape=input_shape[1])

    if include_sar_input:
        x = Concatenate(axis=1)([input_opt, input_sar])
    else:
        x = input_opt

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x)

    # main body of network as succession of resblocks
    for i in range(num_layers):
        x = resBlock(x, feature_size, kernel_size=[3, 3])

    # One more convolution
    x = Conv2D(input_shape[0][0], (3, 3), kernel_initializer='he_uniform', padding='same')(x)

    # Add first layer (long skip connection)
    x = Add()([x, input_opt])

    if use_cloud_mask:
        # the hacky trick with global variables and with lambda functions is needed to avoid errors when
        # pickle saving the model. Tensors are not pickable.
        # This way, the Lambda function has no special arguments and is "encapsulated"

        shape_n = tf.shape(input_opt)

        def concatenate_array(x):
            global shape_n
            return K.concatenate([x, K.zeros(shape=(batch_per_gpu, 1, shape_n[2], shape_n[3]))], axis=1)

        x = Concatenate(axis=1)([x, input_opt])

        x = Lambda(concatenate_array)(x)

    model = Model(inputs=[input_opt, input_sar], outputs=x)

    return model, shape_n
