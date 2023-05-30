import torch

# Let's assume idx is your tensor of indices
idx = torch.tensor([1, 2, 1, 3, 2, 3, 1])

unique_elements = torch.unique(idx, return_counts=True)

result = []
for element in unique_elements[0]:
    element_indices = torch.where(idx == element)[0]
    result.append(element_indices)

# Padding with -1s for tensors with fewer indices
max_len = max([t.size(0) for t in result])
result_padded = torch.stack([torch.cat((t, torch.full((max_len - t.size(0),), -1, dtype=torch.long))) for t in result])

print(result_padded[0])
