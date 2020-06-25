import torch

# Initialize vectors
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])

print(x*y)

# Zero matrix, define the size ([rows, columns])
x = torch.zeros([2,5])
print(x)

# randomly assigned values
y = torch.rand([2,5])
print(y)

# flatten/ reshape into column vector
# be sure to reassign the matrix to the view command
y = y.view([1,10])
print(y)

