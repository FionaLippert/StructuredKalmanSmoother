import torch
import torch.nn.functional as F

def conv2matrix(kernel, img_shape, zero_padding=1):
    """
    Converts a convolution kernel to the corresponding Toeplitz matrix, using zero padding to retain original size.
    Note that when multiplying the flattened input image with the Toeplitz matrix, no explicit zero padding is required.
    :param kernel: (out_channels, in_channels, kernel_height, kernel_width)
    :param img_shape: (in_channels, img_height, img_width)
    :param zero_padding: width of zero padding
    :return: Toeplitz matrix M (out_channels * img_height * img_width, in_channels * img_height * img_width)
    """

    # TODO: write test case for this!

    assert img_shape[0] == kernel.shape[1]
    assert len(img_shape[1:]) == len(kernel.shape[2:])

    padded_img_shape = (img_shape[0], img_shape[1] + 2 * zero_padding, img_shape[2] + 2 * zero_padding)

    M = torch.zeros((kernel.shape[0],
                     *((torch.tensor(padded_img_shape[1:]) - torch.tensor(kernel.shape[2:])) + 1),
                     *padded_img_shape))

    for i in range(M.shape[1]):
        for j in range(M.shape[2]):
            M[:, i, j, :, i:i + kernel.shape[2], j:j + kernel.shape[3]] = kernel

    M = M.flatten(0, len(kernel.shape[2:])).flatten(1)

    pads = [zero_padding] * 4
    mask = F.pad(torch.ones(img_shape), pads).flatten()
    mask = mask.bool()
    M = M[:, mask]

    return M

def block2flat(block_matrix):

    """
    Flatten a block matrix with separate block dimensions
    :param block_matrix: (..., outer_height, outer_width, block_height, block_width)
    :return: flat_matrix (..., outer_height * block_height, outer_width * block_width)
    """

    h, w, bh, bw = block_matrix.shape[-4:]

    return block_matrix.transpose(-3, -2).reshape(-1, h * bh, w * bw)

def flat2block(flat_matrix, block_height, block_width):

    """
    Separate flat matrix into blocks
    :param flat_matrix: (..., outer_height * block_height, outer_width * block_width)
    :return: block_matrix (..., outer_height, outer_width, block_height, block_width)
    """

    assert flat_matrix.shape[-2] % block_height == 0 and flat_matrix.shape[-1] % block_width == 0

    outer_height = flat_matrix.shape[-2] // block_height
    outer_width = flat_matrix.shape[-1] // block_width

    return flat_matrix.reshape(-1, outer_height, outer_width, block_height, block_width).transpose(-3, -2)
