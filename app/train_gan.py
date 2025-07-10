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


# ----------------------------
# 학습 설정
# ----------------------------
def train_gan(real_data_path="app/data/lotto_one_hot.npy",
              noise_dim=20, batch_size=64, num_epochs=3000, lr=0.0002, device="cpu"):

    # 1. 데이터 불러오기
    real_data = np.load(real_data_path).astype(np.float32)
    dataset = TensorDataset(torch.tensor(real_data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 모델 초기화
    G = Generator(noise_dim=noise_dim).to(device)
    D = Discriminator().to(device)

    # 3. 옵티마이저
    g_opt = torch.optim.Adam(G.parameters(), lr=lr)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr)

    # 4. 손실 함수
    criterion = nn.BCELoss()

    # 5. 학습 루프
    for epoch in tqdm(range(num_epochs)):
        for real_batch, in dataloader:
            real_batch = real_batch.to(device)

            # ----------------
            # 1. Discriminator 학습
            # ----------------
            batch_size = real_batch.size(0)
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_batch = G(noise).detach()  # fake 데이터

            # 라벨
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            # D loss
            d_real_loss = criterion(D(real_batch), real_labels)
            d_fake_loss = criterion(D(fake_batch), fake_labels)
            d_loss = d_real_loss + d_fake_loss

            D.zero_grad()
            d_loss.backward()
            d_opt.step()

            # ----------------
            # 2. Generator 학습
            # ----------------
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_batch = G(noise)
            fake_labels = torch.ones((batch_size, 1)).to(device)  # G는 D를 속이고 싶음

            g_loss = criterion(D(fake_batch), fake_labels)

            G.zero_grad()
            g_loss.backward()
            g_opt.step()

        # 🔁 로그 출력 (10 epoch마다)
        if (epoch + 1) % 10 == 0:
            print(f"[{epoch+1}/{num_epochs}] D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}")

    # ✅ 최종 Generator 저장
    torch.save(G.state_dict(), "generator.pth")
    print("✅ Generator 모델 저장 완료: generator.pth")

    return G

if __name__ == "__main__":
    train_gan(device="cuda" if torch.cuda.is_available() else "cpu")