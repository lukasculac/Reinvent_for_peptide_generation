import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from sympy import sequence
from torch.utils.data import Dataset

from utils import Variable

class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, max_length=50):
        self.special_tokens = ['EOS', 'GO']
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

        self.chars = self.amino_acids + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length


    def encode(self, sequence):
        """Takes a peptide sequence and encodes to array of indices"""

        matrix = np.zeros(len(sequence), dtype=np.float32)
        for i, char in enumerate(sequence):
            matrix[i] = self.vocab[char]
        return matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the peptide seuence"""
        amino_acids = []

        # Convert tensor to CPU and then to regular numbers
        if torch.is_tensor(matrix):
            matrix = matrix.cpu().numpy()

        for i in matrix:
            if i == self.vocab['EOS']: break
            amino_acids.append(self.reversed_vocab[i])
        return "".join(amino_acids)

    def tokenize(self, sequence):
        """Takes a peptide sequence and return a list of amino acids + EOS"""
        return [aa for aa in sequence] + ['EOS']

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)

class PeptideData(Dataset):
    """Custom PyTorch Dataset for peptide sequences with labels"""

    def __init__(self, fname, voc):
        self.voc = voc
        self.sequences = []
        self.labels = []

        # Load peptides and labels from CSV
        with open(fname, 'r') as f:
            next(f) #skip first row
            for line in f:
                data = line.strip().split(',')
                sequence = data[0]
                label = int(data[1])
                self.sequences.append(sequence)
                self.labels.append(int(label))

    def __getitem__(self, i):
        """Returns a single encoded sequence and its label"""
        sequence = self.sequences[i]
        tokenized = self.voc.tokenize(sequence)
        encoded = self.voc.encode(tokenized)
        return Variable(encoded), self.labels[i]

    def __len__(self):
        return len(self.sequences)



    @classmethod
    def collate_fn(cls, arr):
        """Creates batches of sequences with padding"""
        #separate sequences and labels
        sequences = [item[0] for item in arr]
        labels = [item[1] for item in arr]

        # Pad sequences to max length in batch
        max_length = max([seq.size(0) for seq in sequences])
        collated_seqs = Variable(torch.zeros(len(arr), max_length))
        for i, seq in enumerate(sequences):
            collated_seqs[i, :seq.size(0)] = seq

        return collated_seqs, torch.tensor(labels)

class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""
    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory)>self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key = lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory)<n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores/np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[xx]' as one token."""
    regex = '(\[[^\[\]]{1,6}\])'
    smiles = replace_halogen(smiles)
    char_list = re.split(regex, smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('EOS')
    return tokenized

def canonicalize_smiles_from_file(fname):
    """Reads a SMILES file and returns a list of RDKIT SMILES"""
    with open(fname, 'r') as f:
        smiles_list = []
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("{} lines processed.".format(i))
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if filter_mol(mol):
                smiles_list.append(Chem.MolToSmiles(mol))
        print("{} SMILES retrieved".format(len(smiles_list)))
        return smiles_list

def filter_mol(mol, max_heavy_atoms=50, min_heavy_atoms=10, element_list=[6,7,8,9,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False

def write_smiles_to_file(smiles_list, fname):
    """Write a list of SMILES to a file."""
    with open(fname, 'w') as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")

def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in smiles_list:
        tokenized = tokenize(smiles)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    return smiles_list_valid

def filter_file_on_chars(smiles_fname, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    with open(smiles_fname, 'r') as f:
        for line in f:
            smiles.append(line.split()[0])
    print(smiles[:10])
    chars = []
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print(chars)
    valid_smiles = filter_on_chars(smiles, chars)
    with open(smiles_fname + "_filtered", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def combine_voc_from_files(fnames):
    """Combine two vocabularies"""
    chars = set()
    for fname in fnames:
        with open(fname, 'r') as f:
            for line in f:
                chars.add(line.split()[0])
    with open("_".join(fnames) + '_combined', 'w') as f:
        for char in chars:
            f.write(char + "\n")

def construct_vocabulary(smiles_list):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open('data/Voc', 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print("Reading smiles...")
    smiles_list = canonicalize_smiles_from_file(smiles_file)
    print("Constructing vocabulary...")
    voc_chars = construct_vocabulary(smiles_list)
    write_smiles_to_file(smiles_list, "data/mols_filtered.smi")
