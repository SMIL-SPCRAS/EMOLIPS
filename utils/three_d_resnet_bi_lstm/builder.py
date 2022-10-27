from . import three_D_resnet
from .kernel import get_kernel_to_name


def build_three_d_resnet_18(input_shape, output_shape, output_activation, regularizer=None,
                            squeeze_and_excitation=False, kernel_name='3D'):
    """Return a customizable resnet_18.

    :param input_shape: The input shape of the network as (frames, height, width, channel)
    :param output_shape: The output shape. Dependant on the task of the network.
    :param output_activation: Define the used output activation. Also depends on the task of the network.
    :param regularizer: Defines the regularizer to use. E.g. "l1" or "l2"
    :param squeeze_and_excitation:Activate or deactivate SE-Paths.
    :param kernel_name:
    :return: The built ResNet-18
    """
    conv_kernel = get_kernel_to_name(kernel_name)
    return three_D_resnet.ThreeDConvolutionResNet(input_shape, output_shape, output_activation, (2, 2, 2, 2),
                                                  regularizer, squeeze_and_excitation, kernel=conv_kernel)