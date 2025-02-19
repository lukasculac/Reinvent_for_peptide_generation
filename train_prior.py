import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_structs import Vocabulary, PeptideData
from model import RNN
from utils import Variable, decrease_learning_rate

def pretrain(restore_from=None):
    """Trains the Prior RNN on  peptide sequences"""

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create Vocabulary class instance
    voc = Vocabulary()

    # Create a Dataset from a FASTA file
    pepdata = PeptideData("data/dataset.csv", voc)
    data = DataLoader(pepdata, batch_size=128, shuffle=True, drop_last=True,
                      collate_fn=pepdata.collate_fn)

    # Initialize the model and move it to the selected device
    Prior = RNN(voc).to(device)

    # Can restore from a saved RNN
    if restore_from:
        Prior.rnn.load_state_dict(torch.load(restore_from, map_location=device))

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=0.001)
    for epoch in range(1, 6):
        #track total loss for the epoch
        epoch_loss = 0
        for step, batch in tqdm(enumerate(data), total=len(data)):

            #Unpack sequences and labels
            seqs, labels = batch
            seqs = seqs.long().to(device)

            # Calculate loss
            log_p, _ = Prior.likelihood(seqs)
            loss = -log_p.mean()
            epoch_loss += loss.item()

            # Calculate gradients and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 500 steps we decrease learning rate and print some information
            if step % 500 == 0 and step != 0:
                decrease_learning_rate(optimizer, decrease_by=0.03)
                tqdm.write("*" * 50)
                tqdm.write("Epoch {:3d}   step {:3d}    loss: {:5.2f}\n".format(epoch, step, loss.item()))

                #Sample some sequences
                seqs, likelihood, _ = Prior.sample(128)
                valid = 0
                for i, seq in enumerate(seqs.cpu().numpy()):
                    peptide = voc.decode(seq)
                    # Validate peptide sequence (only contains valid amino acids)
                    if all(aa in voc.amino_acids for aa in peptide):
                        valid += 1
                    if i < 5:
                        tqdm.write(f"Generated peptide: {peptide}")

                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")

                # Save the Prior
                torch.save(Prior.rnn.state_dict(), "data/Prior_checkpoint.ckpt")

        # Print epoch summary
        avg_loss = epoch_loss / len(data)
        print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")

        # Save the Prior after each epoch
        torch.save(Prior.rnn.state_dict(), f"data/Prior_epoch_{epoch}.ckpt")
if __name__ == "__main__":
    pretrain()