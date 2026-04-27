import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

data = [[0.0, 0], [0.1, 0], [0.2, 0], [0.3, 0], [0.4, 1], [0.5, 1], [0.6, 1], [0.7, 1], [0.8, 1], [0.9, 0], [1.0, 0]]
# X = np.array(data)[:, 0]
# y = np.array(data)[:, 1]
# plt.scatter(X, y)
# plt.show()

ten = torch.Tensor(data)
feature = (ten[:, 0]).unsqueeze(dim=1)
target = (ten[:, 1]).unsqueeze(dim=1)

n = 2 # 은닉층 노드 수
model = nn.Sequential(
    nn.Linear(1, n),
    nn.ReLU(),
    nn.Linear(n, 1),
)

# init.kaiming_normal_(model[0].weight, nonlinearity='relu')
# init.zeros_(model[0].bias)

# init.xavier_normal_(model[2].weight)
# init.zeros_(model[2].bias)

optimizer = optim.SGD(model.parameters(), lr=0.0223)
criterion = nn.BCEWithLogitsLoss()

model.train()
epochs = 50000
acc = 0
for epoch in range(epochs):
    optimizer.zero_grad()
    
    pred = model(feature)
    loss = criterion(pred, target)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 999:
        y = (pred > 0).to(torch.int)
        acc = (y == target).float().mean() * 100
        print(f'[{epoch + 1}epoch] acc: {acc:.2f}%, loss: {loss:.2f}')

    if acc == 100:
        break


# 학습 종료 후: sigmoid 전(logit) + sigmoid 후(probability) 둘 다 시각화

model.eval()

with torch.no_grad():
    xs = torch.linspace(0, 1, 500).unsqueeze(1)

    logits = model(xs)              # sigmoid 전 값
    probs = torch.sigmoid(logits)   # sigmoid 후 확률값

# numpy 변환
x_plot = xs.squeeze().numpy()
logit_plot = logits.squeeze().numpy()
prob_plot = probs.squeeze().numpy()

X = feature.squeeze().numpy()
Y = target.squeeze().numpy()

plt.figure(figsize=(11,8))

# -------------------
# 위: sigmoid 전 값
# -------------------
plt.subplot(2,1,1)
plt.plot(x_plot, logit_plot, linewidth=2, label='Logit (before sigmoid)')
plt.axhline(0, linestyle=':', linewidth=1)   # 0 기준선 = 분류 경계
plt.scatter(X, Y*0, s=50, alpha=0.4, label='x positions')
plt.title("Raw Output (Logits)")
plt.ylabel("logit")
plt.legend()
plt.grid(True, alpha=0.3)

# -------------------
# 아래: sigmoid 후 값
# -------------------
plt.subplot(2,1,2)
plt.plot(x_plot, prob_plot, linewidth=2, label='Probability')
plt.scatter(X, Y, s=70, zorder=3, label='Train Data')
plt.axhline(0.5, linestyle=':', linewidth=1)
plt.title("Sigmoid Output")
plt.xlabel("x")
plt.ylabel("probability")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()