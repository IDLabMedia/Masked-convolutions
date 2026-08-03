import torch
from torch import nn
import inspect
import torch.nn.functional as F

def copy_mask(mask):
    """
    Returns a detached copy of the mask tensor for independent use.
    """
    return mask.clone().detach()

def mask_to_coords(distraction_cover):
    """
    Convert a binary mask to coordinates of the smallest rectangle that covers the mask.
    Input:
        distraction_cover: Bx1xHxW with 1s in the mask area and 0s in the rest
    Output:
        ((t, l), (h, w)): (top, left), (height, width)
    """

    # distraction_cover is Bx1xHxW with 1s in the mask area and 0s in the rest
    if distraction_cover is not None:
        mask = distraction_cover.clone().detach()
        rows_ = mask.sum(dim=(2))
        cols_ = mask.sum(dim=(3))
        h, l = rows_.max(dim=(2)) # shape [B, C_in]
        h = h[:, 0] # shape [B]
        l = l[:, 0]
        w, t = cols_.max(dim=(2)) # shape [B, C_in]
        w = w[:, 0] # shape [B]
        t = t[:, 0]

        coords = ((int(t[0]), int(l[0])), (int(h[0]), int(w[0]))) # assuming batch size = 1

    else:
        coords = ((0, 0), (0, 0)) # no mask

    return coords


def create_mask(distraction_cover, x=None):
    """
    Creates empty mask if distraction_cover is None
    """
    if distraction_cover is None:
        B, C, H, W = x.shape
        # print("Warning: distraction_cover is None, creating empty mask.")
        # print("x shape:", x.shape)
        mask = torch.zeros((B, 1, H, W), device=x.device, dtype=torch.float)
    else:
        mask = distraction_cover


    return mask


class MaskedConv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input of arbitrary size, but ignores the masked region.
    Input:
        x: input image, shape (B, C_in, H, W)
        mask: binary mask, shape (B, 1, H, W), with 1s in the masked region and 0s in the rest
    """
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        ## Not stored as convolution layer because not trainable
        k_r, k_c = self.kernel_size
        p_r, p_c = self.padding

        dilated = False
        if self.dilation != 1 and self.dilation != (1, 1):
            dilated = True


        if dilated:
            e_k_r = self.dilation[0] * (k_r - 1) + 1
            e_k_c = self.dilation[1] * (k_c - 1) + 1
            print(f"Warning: Dilation {self.dilation} updating kernel ({k_r}, {k_c}) to effective kernel ({e_k_r}, {e_k_c}).")
            # 0 -> 0
            # 1 - 8 -> 1
            # 9 - 16 -> 2
            # ...
            e_p_r = (p_r - 1) // self.dilation[0] + 1
            e_p_c = (p_c - 1) // self.dilation[1] + 1


        if not dilated and (  p_r >= (k_r + 1) // 2 or   p_c >= (k_c + 1) // 2):
            raise ValueError(f"Padding {self.padding} is too large for kernel size {self.kernel_size} in MaskedConv2d.")
        elif   dilated and (e_p_r >= (k_r + 1) // 2 or e_p_c >= (k_c + 1) // 2):
            raise ValueError(f"Effective padding ({e_p_r}, {e_p_c}) is too large for effective kernel size ({e_k_r}, {e_k_c}) in MaskedConv2d.")

        # make a fixed kernel
        # kernel only 1 where not padded
        # -> if output is 1, means non-padded regions is outside the image, aka invalid kernel window
        kernel = torch.zeros((1, 1, k_r, k_c))
        if not dilated:
            kernel[0, 0,   p_r:k_r -   p_r,   p_c:k_c -   p_c] = 1.0
        else:
            kernel[0, 0, e_p_r:k_r - e_p_r, e_p_c:k_c - e_p_c] = 1.0

        # register as buffer and non persistent -> not trainable, not in optimizer, and not saved with state_dict
        self.register_buffer("mask_kernel", kernel, persistent=False)

    
    def convolve_mask(self, x):
        return F.conv2d(
            x,
            self.mask_kernel,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation
        )


    def forward(self, x, mask=None):
        if mask is None:
            return super(MaskedConv2d, self).forward(x)

        #### CHECK MASK DIMENSIONS
        
        if mask.shape[0] != x.shape[0]:
            raise ValueError(f"Mask batch size {mask.shape[0]} does not match input batch size {x.shape[0]}.")
        
        if mask.shape[2] != x.shape[2] or mask.shape[3] != x.shape[3]:
            # print(f"Warning: Mask shape {mask.shape} does not match input shape {x.shape}")
            raise ValueError(f"Mask shape {mask.shape} does not match input shape {x.shape}")

            # ## Option 1: resize to match input size (means that mask will be pulled to correct size)
            # ## !!! need to be binarized after interpolation
            # mask = F.interpolate(mask, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            # mask = (mask > 0.5).float()

            ## Option 2: just ignore the mask (mask to small to be useful)
            mask = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device, dtype=torch.float)


        #### PROCESS

        B, C_in, H, W = x.shape

        # apply mask and conv2d
        mask_expanded = mask.expand(-1, C_in, -1, -1)                  # Expand mask to match input channels
        x_masked = x * (1 - mask_expanded)                             # Apply mask to input image (zero out masked pixels)
        x_masked_out = super(MaskedConv2d, self).forward(x_masked)     # Perform convolution on masked input

        # update mask
        mask_out = self.convolve_mask(mask)                            # Apply convolve_mask to the binary mask
        mask_out = (mask_out > 0.5).float()                            # Binarize mask to 0 or 1

        # zero invalid regions
        mask_out_expanded = mask_out.expand(-1, x_masked_out.shape[1], -1, -1)
        x_out = x_masked_out * (1 - mask_out_expanded)


        return x_out, mask_out

    

class MaskedAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    Applies a 2D adaptive average pooling over an input
    of arbitrary size, but ignores the masked region.
    Input:
        x: input image, shape (B, C_in, H, W)
        mask: binary mask, shape (B, 1, H, W), with 1s in the masked region and 0s in the rest
    Note: if all pixels in a pooling region are masked, the output is set to 0.0
    """
    def __init__(self, *args, **kwargs):
        super(MaskedAdaptiveAvgPool2d, self).__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        if mask is None:
            return super(MaskedAdaptiveAvgPool2d, self).forward(x)

        B, C_in, H, W = x.shape

        # make sure mask is binary
        mask = mask.bool()

        # apply mask and avgpool (sum / # el in cell)
        mask_expanded = mask.expand(-1, C_in, -1, -1)                       # Expand mask to match input channels
        x_masked = x.masked_fill(mask_expanded, 0.0)                        # Apply mask to input image (0 for masked pixels)
        avg_pool = super(MaskedAdaptiveAvgPool2d, self).forward(x_masked)   # Perform adaptive avg pooling on masked input (sum / # el in cell)
        inv_mask_expanded = (~mask_expanded).float()                        # Inverted mask (1s where unmasked) and expand to match input channels

        # count unmasked pixels in each pooling region (# valid / # el in cell)
        count = super(MaskedAdaptiveAvgPool2d, self).forward(inv_mask_expanded)  # Count of unmasked pixels in each pooling sample (# valid / # el in cell)
        count_nozeros = count.clone()
        count_nozeros[count_nozeros == 0] = 1.0                             # Avoid division by zero
        
        # calculate average (sum / # valid = (sum / # el in cell) / (# valid / # el in cell))
        avg_pool = avg_pool / count_nozeros                                  # Divide sum by count of unmasked pixels to get average (sum / # valid)

        # replace fully masked samples with default 0.0
        fully_masked = (count == 0)                                         # Identify fully masked pooling regions
        avg_pool = avg_pool.masked_fill(fully_masked, 0.0)

        return avg_pool


class MaskedAdaptiveMaxPool2d(nn.AdaptiveMaxPool2d):
    """
    Applies a 2D adaptive max pooling over an input
    of arbitrary size, but ignores the masked region.
    Input:
        x: input image, shape (B, C_in, H, W)
        mask: binary mask, shape (B, 1, H, W), with 1s in the masked region and 0s in the rest
    Note: if all pixels in a pooling region are masked, the output is set to 0.0
    """
    def __init__(self, *args, **kwargs):
        super(MaskedAdaptiveMaxPool2d, self).__init__(*args, **kwargs)

    def forward(self, x, mask=None):
        if mask is None:
            return super(MaskedAdaptiveMaxPool2d, self).forward(x)
        
        B, C_in, H, W = x.shape

        # make sure mask is binary
        mask = mask.bool()

        # apply mask and maxpool
        mask_expanded = mask.expand(-1, C_in, -1, -1)                       # Expand mask to match input channels
        x_masked = x.masked_fill(mask_expanded, float("-inf"))              # Apply mask to input image (-inf for masked pixels)
        max_pool = super(MaskedAdaptiveMaxPool2d, self).forward(x_masked)   # Perform adaptive max pooling on masked input

        # check for fully masked blocks
        inv_mask_expanded = (~mask_expanded).float()                        # Inverted mask (1s where unmasked) and expand to match input channels
        any_unmasked = super(MaskedAdaptiveMaxPool2d, self).forward(inv_mask_expanded)  # Check if there is atleast one unmasked pixel in each pooling sample
        fully_masked = (any_unmasked == 0)                                  # Identify fully masked pooling regions

        # replace fully masked samples with default 0.0
        max_pool = max_pool.masked_fill(fully_masked, 0.0)

        return max_pool


class MaskedSequential(nn.Module):
    """
    A sequential container for masked modules.

    Modules will be added to it in the order they are passed in the constructor.
    During the forward pass, if a module is a masked module (its class name starts with 'Masked'),
    it will receive both the input and the mask. Otherwise, it will only receive the input.
    """
    def __init__(self, *args):
        super(MaskedSequential, self).__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input, mask):
        if mask is None:
            for module in self._modules.values():
                input = module(input)
            return input
        
        for module in self._modules.values():
            # only if masked module pass mask (name starts with Masked)
            if module.__class__.__name__.startswith('Masked'):
                input, mask = module(input, mask)
            else:
                input = module(input)
        return input, mask