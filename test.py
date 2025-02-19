from data_structs import Vocabulary

voc = Vocabulary()

# Encode a peptide sequence
sequence = "ACDE"
tokens = list(sequence) + ['EOS']  # Add EOS token
encoded = voc.encode(tokens)
print(encoded)  # Output: [0, 1, 2, 3, 20] (assuming A=0, C=1, D=2, E=3, EOS=20)

# Decode an encoded sequence
decoded = voc.decode(encoded)
print(decoded)  # Output: "ACDE"