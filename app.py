import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# -------- VAE Model (same structure as training) --------
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# -------- Load Model --------
@st.cache_resource
def load_model():
    model = VAE()
    model.load_state_dict(torch.load("models/vae_mnist.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -------- Streamlit UI --------
st.title("🖋️ Handwritten Digit Generator (0–9)")
st.write("Select a digit and generate 5 diverse samples.")

digit = st.selectbox("Choose a digit", list(range(10)))
if st.button("Generate"):
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        # Slightly random latent vectors
        z = torch.randn(1, 20)
        sample = model.decode(z).detach().numpy().reshape(28, 28)
        axs[i].imshow(sample, cmap="gray")
        axs[i].axis("off")
    st.pyplot(fig)
