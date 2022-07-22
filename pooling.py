import random

import torch
import torch.nn as nn
m = nn.AdaptiveAvgPool2d((1,5))
torch.manual_seed(10)
input = torch.randn(1, 3, 8, 9)
print (input)
output = m(input)
print (output)

preds = torch.softmax(output, dim=1)
print ('preds_shape',preds.shape,preds)


pred, class_idx = torch.max(preds, dim=1)
print ("class_idx",class_idx,class_idx.shape,pred)

row_max, row_idx = torch.max(pred, dim=1)
print ('row',row_max,row_idx)
col_max, col_idx = torch.max(row_max, dim=1)
print ('col',col_max,col_idx)
predicted_class = class_idx[0, row_idx[0, col_idx], col_idx]

# Print top predicted class
print('Predicted Class : ',  predicted_class)

