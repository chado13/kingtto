import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ----------------------------
# 모델 정의
# ----------------------------
class Generator(nn.Module):
    def __init__(self, noise_dim=20, output_dim=45):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim=45):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def main():
    G = load_generator()
    res = generate_lotto_numbers(G, k=6, noise_dim=20, num_samples=5, device="cpu")
    for i, r in enumerate(res):
        print(f"{i} 번째 추천 번호: {r}")


def load_generator(model_path="app/model/generator.pth", noise_dim=20, device="cpu"):
    G = Generator(noise_dim=noise_dim).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    return G


def generate_lotto_numbers(G, k=6, noise_dim=20, num_samples=5, device="cpu"):
    noise = torch.randn(num_samples, noise_dim).to(device)
    with torch.no_grad():
        outputs = G(noise)

    # 확률 벡터 → 상위 k개 번호로 변환
    lotto_sets = []
    for vec in outputs:
        topk = torch.topk(vec, k=k).indices + 1
        lotto_sets.append(sorted(topk.cpu().numpy().tolist()))

    return lotto_sets


if __name__ == "__main__":
    main()