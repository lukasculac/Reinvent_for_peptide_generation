import torch
from torch.utils.data import DataLoader
import pickle
from rdkit import Chem
from rdkit import rdBase
from tqdm import tqdm

from data_structs import MolData, Vocabulary
from model import RNN
from utils import Variable, decrease_learning_rate
rdBase.DisableLog('rdApp.error')

def pretrain(restore_from=None):
    """Trains the Prior RNN"""

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Read vocabulary from a file
    voc = Vocabulary(init_from_file="data/Voc_peptides")

    # Create a Dataset from a SMILES file
    moldata = MolData("data/mols_filtered.smi", voc)
    data = DataLoader(moldata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=MolData.collate_fn)

    # Initialize the model and move it to the selected device
    Prior = RNN(voc).to(device)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from, map_location=device))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)
    for epoch in range(1, 6):
        for step, batch in tqdm(enumerate(data), total=len(data)):

            # Sample from DataLoader and move batch to GPU
            seqs = batch.long().to(device)

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = -log_p.mean()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                    if i < 5:
                        tqdm.write(smile)
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")
                torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

        # Save the Prior
        torch.save(Prior.rnn.state_dict(), "data/Prior.ckpt")

if __name__ == "__main__":
    pretrain()