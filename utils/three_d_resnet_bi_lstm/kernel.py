from tensorflow import keras

class Conv3DBlock(keras.layers.Layer):
    """
    Convenient layer to create Conv3D -> Batch Normalization -> ReLU blocks faster
    """
    def __init__(self, kernel_number, kernel_size, regularizer=None, strides=1, use_bn=False, padding='same',
                 use_activation=True,  **kwargs):
        super(Conv3DBlock, self).__init__(**kwargs)
        self.custom_conv_3d = keras.Sequential()
        self.custom_conv_3d.add(keras.layers.Conv3D(kernel_number, kernel_size, strides, padding=padding,
                                                    use_bias=not use_bn, kernel_regularizer=regularizer))
        if use_bn:
            self.custom_conv_3d.add(keras.layers.BatchNormalization())
        if use_activation:
            self.custom_conv_3d.add(keras.layers.ReLU())

    def __call__(self, inputs, training=None):
        return self.custom_conv_3d(inputs, training=training)

class ThreeD(keras.layers.Layer):
    def __init__(self,  kernel_number, kernel_size, strides, padding, use_bn, kernel_regularizer,
                 use_activation=True, **kwargs):
        super(ThreeD, self).__init__(**kwargs)
        self.three_d = Conv3DBlock(kernel_number, (1, kernel_size, kernel_size),
                                   kernel_regularizer, (1, strides, strides), use_bn, padding, use_activation=use_activation)

    def __call__(self, inputs, training=None):
        return self.three_d(inputs, training=training)

def get_kernel_to_name(kernel_type):
    if kernel_type == '3D':
        return ThreeD
