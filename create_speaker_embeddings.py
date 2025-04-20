# create_speaker_embeddings.py
import torch
from datasets import load_dataset

# Load valid speaker embeddings
dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = {
    "bdl": dataset[0]["xvector"],  # Male voice
    "slt": dataset[1]["xvector"],  # Female voice
    "rms": dataset[2]["xvector"],  # Male voice
    "clb": dataset[3]["xvector"]   # Female voice
}

torch.save(speaker_embeddings, "speaker_embeddings.pt")
print("Created speaker embeddings with voices: bdl, slt, rms, clb")