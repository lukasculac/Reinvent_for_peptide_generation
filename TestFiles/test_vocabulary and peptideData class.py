from data_structs import Vocabulary, PeptideData
import torch
from torch.utils.data import DataLoader

# Initialize vocabulary and dataset
voc = Vocabulary()
dataset = PeptideData('../data/dataset.csv', voc)

# Test single item
print("Testing single item retrieval:")
sequence, label = dataset[0]
print(f"First sequence encoded: {sequence}")
print(f"First sequence decoded: {voc.decode(sequence)}")
print(f"First label: {label}")
print(f"Original sequence length: {len(dataset.sequences[0])}")
print(f"Encoded sequence length: {len(sequence)}")

# Test batch creation
print("\nTesting batch creation:")
loader = DataLoader(dataset,
                   batch_size=2,
                   shuffle=False,
                   collate_fn=PeptideData.collate_fn)

# Get first batch
batch_sequences, batch_labels = next(iter(loader))
print(f"Batch shape: {batch_sequences.shape}")
print(f"Labels shape: {batch_labels.shape}")
print(f"Batch labels: {batch_labels}")

# Decode first two sequences in batch
print("\nDecoding batch sequences:")
for i in range(len(batch_sequences)):
    decoded = voc.decode(batch_sequences[i])
    print(f"Original sequence {i}: {dataset.sequences[i]}")
    print(f"Decoded sequence {i}: {decoded}")

"""Testing single item retrieval:
First sequence encoded: tensor([ 6., 15.,  3.,  5., 16.,  4., 15., 15.,  2., 19., 15.,  8., 19.,  9.,
         2., 15., 14., 14.,  0.,  8.,  2.,  4., 17., 13., 18.,  9., 10., 15.,
        16., 20.], device='cuda:0')
First sequence decoded: HSEGTFSSDYSKYLDSRRAKDFVQWLMST
First label: 0
Original sequence length: 29
Encoded sequence length: 30

Testing batch creation:
Batch shape: torch.Size([2, 49])
Labels shape: torch.Size([2])
Batch labels: tensor([0, 0])

Decoding batch sequences:
Original sequence 0: HSEGTFSSDYSKYLDSRRAKDFVQWLMST
Decoded sequence 0: HSEGTFSSDYSKYLDSRRAKDFVQWLMST
Original sequence 1: MARFPEAEARLLNVKICMKCNARNAIRATSCRKCGSDELRAKSKERKA
"""