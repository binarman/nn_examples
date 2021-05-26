#!/usr/bin/python3

import torch

filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
bias = torch.randn(8, dtype=torch.float)

scale, zero_point = 1.0, 0
dtype_inputs = torch.quint8
dtype_filters = torch.qint8

# q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
q_filters = torch.quantize_per_channel(filters, torch.tensor([scale]*8), torch.tensor([zero_point]*8), 0, dtype_filters)
#q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
q_inputs = torch.quantize_per_channel(inputs, torch.tensor([scale]*4), torch.tensor([zero_point]*4), 1, dtype_inputs)
qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
