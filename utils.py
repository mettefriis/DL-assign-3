################################################################################
# MIT License
#
# Copyright (c) 2022
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Autumn 2022
# Date Created: 2022-11-25
################################################################################

import torch
from torchvision.utils import make_grid
from torch.distributions import Normal
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    epsilon = torch.randn_like(std) #sample random epislon of from N(0,1) with shape like std.
    
    z = std*epsilon+mean 

    #######################
    # END OF YOUR CODE    #
    #######################
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See the definition of the regularization loss in Section 1.4 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    #we note that exp(2*log(std))=exp(log(std))^2=std^2
    std = torch.exp(log_std)

    #per d
    KLD_d = 0.5 * (std**2 + mean**2 - 1 - 2 * log_std)
    #sum all d:
    KLD=torch.sum(KLD_d, dim=-1)
    #######################
    # END OF YOUR CODE    #
    #######################
    return KLD


def elbo_to_bpd(elbo, img_shape):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    _, channels, height, width = img_shape #extract dimensions, excluding batch
    num_pixels = channels * height * width #calculate product
    
    bpd = elbo/num_pixels * torch.log2(torch.exp(torch.tensor(1.0)))
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return bpd


@torch.no_grad()
def visualize_manifold(decoder, grid_size=20):
    """
    Visualize a manifold over a 2 dimensional latent space. The images in the manifold
    should represent the decoder's output means (not binarized samples of those).
    Inputs:
        decoder - Decoder model such as LinearDecoder or ConvolutionalDecoder.
        grid_size - Number of steps/images to have per axis in the manifold.
                    Overall you need to generate grid_size**2 images, and the distance
                    between different latents in percentiles is 1/grid_size
    Outputs:
        img_grid - Grid of images representing the manifold.
    """

    ## Hints:
    # - You can use the icdf method of the torch normal distribution  to obtain z values at percentiles.
    # - Use the range [0.5/grid_size, 1.5/grid_size, ..., (grid_size-0.5)/grid_size] for the percentiles.
    # - torch.meshgrid might be helpful for creating the grid of values
    # - You can use torchvision's function "make_grid" to combine the grid_size**2 images into a grid
    # - Remember to apply a softmax after the decoder

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    
    #generate perccentiles, evenly spaced
    percentiles = torch.linspace(0.5 / grid_size, (grid_size - 0.5) / grid_size, grid_size)
    normal = Normal(0, 1) #deifne normal distribution
    z_values = normal.icdf(percentiles) # Shape: [grid_size] #use function to otbain z values at the evenly spaced perceniles

    #make a grid of grid_size x grid_size:
    z_rows, z_columns = torch.meshgrid(z_values, z_values, indexing="ij") #z1, z2 specify the row (first) and column (second) cooridnate of entries in the grid
    latent_grid = torch.stack([z_rows.flatten(), z_columns.flatten()], dim=-1)  # Shape: [grid_size**2, 2] Createa a batch format

    decoded_imgs = decoder(latent_grid)
    decoded_imgs = torch.softmax(decoded_imgs, dim=1) #apply softmax

    img_grid = make_grid(decoded_imgs, nrow=grid_size, normalize=True, value_range=(0, 1))
    
    #######################
    # END OF YOUR CODE    #
    #######################

    return img_grid

