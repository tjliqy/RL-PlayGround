import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss()
input = torch.tensor([[0.,1,0],[0,1,0]], requires_grad=True)
target = torch.LongTensor([1,1])
print(input, target)



output = loss(input, target)

# output.backward()

print(output)
print(output.mean())

# m = torch.tensor([1,1,1])
# m = m+1
# print(m)


# pi = torch.tensor([[0.2,0.3,0.5],[0,1,0]])
# print(pi.index_select(0,torch.LongTensor([0,0])))