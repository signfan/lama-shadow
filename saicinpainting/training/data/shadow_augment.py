import torch
import random
import numpy as np

def sample_darkening_params(opt):
    """
        Sample two points, and generate its slope
    """
    assert 0.0 <= opt.xmin_at_y_0 <= 1.0
    assert 0.0 <= opt.xmax_at_y_0 <= 1.0
    assert 0.0 <= opt.ymin_at_x_255 <= 1.0
    assert 0.0 <= opt.ymax_at_x_255 <= 1.0
    assert 1.0 <= opt.slope_max  # opt.slope_max < 1.0 is very `normal`

    while True:
        x1 = random.uniform(opt.xmin_at_y_0, opt.xmax_at_y_0)
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(opt.ymin_at_x_255, opt.ymax_at_x_255)
        a = (y2 - y1) / (x2 - x1)  # Assume slope = const. for all channels
        if a < 1.0:
            break
    return a, x1, y1, x2, y2

def darken(x, opt):
    x = torch.tensor(x)
    a, x1, y1, x2, y2 = sample_darkening_params(opt)
    if opt.x_turb_sigma == 0.0:
        b = y1 - a * x1  # no shift in x_intercept in RGB channel
    else:
        # R and G are usually larger than G and R, respectively.
        # y_g = a * x + b
        x1_G = x1
        mu, sigma = opt.x_turb_mu, opt.x_turb_sigma
        x1_R = x1_G + np.random.normal(mu, sigma)
        x1_B = x1_G - np.random.normal(mu, sigma)

        b = [y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B]
        b = torch.Tensor(b).view(3, 1, 1)
        b = b.repeat(1, x.size(1), x.size(2))

    y = a * x + b
    y = torch.clamp(y, min=0.0, max=1.0)
    y = y.numpy()
    return y