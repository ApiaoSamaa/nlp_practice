# %%
with open('hachidaishu-pos.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines = filter(lambda x: len(x) > 2, lines)
    lines = list(map(lambda x: x.split()[1:], lines))

lines[:10]

# %%

vocab = set([""])
pos_vocab = set([""])

for line in lines:
    for token in line:
        word, pos, _ = token.split("/")
        pos = pos.split(":")[0].split("-")[0]
        vocab.add(word)
        pos_vocab.add(pos)

print(len(vocab), len(pos_vocab))

# %%

word_to_id = { w: i for i, w in enumerate(vocab) }
id_to_word = { i: w for i, w in enumerate(vocab) }

pos_to_id = { w: i for i, w in enumerate(pos_vocab) }
id_to_pos = { i: w for i, w in enumerate(pos_vocab) }

# %%

dataset = []

for line in lines:
    tokens = list(t.split("/")[0] for t in line)
    pos = list(t.split("/")[1].split(":")[0].split("-")[0] for t in line)

    tokens = ["", ""] + tokens + ["", ""]
    pos = ["", ""] + pos + ["", ""]

    for i in range(len(line)):
        dataset.append((
            list(map(lambda t: word_to_id[t], tokens[i:i + 5])),
            pos_to_id[pos[i + 2]],
        ))

print(len(dataset))
train_dataset = dataset[:-1000]
test_dataset = dataset[-1000:]

# %%
import torch
from torch import nn, optim

model = nn.Sequential(
    # LongTensor (batch_size, 5)
    nn.Embedding(len(vocab), 200),
    # FloatTensor (batch_size, 5, 200)
    nn.Flatten(-2),
    # FloatTensor (batch_size, 1000)
    nn.Linear(1000, 200), nn.GELU(),
    nn.Linear(200, 100), nn.GELU(),
    nn.Linear(100, len(pos_vocab)),
)

criteria = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), 3e-4)

# %%
from torch.utils.data import DataLoader

def collate_fn(rows):
    return [
        torch.LongTensor([r[0] for r in rows]),
        torch.LongTensor([r[1] for r in rows]),
    ]

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=100, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=100, collate_fn=collate_fn)

# %%
from tqdm import tqdm

epochs = 5
for epoch in range(epochs):
    bar = tqdm(train_loader, f"train {epoch + 1}")
    for x, y in bar:
        pred = model(x)
        loss = criteria(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bar.set_postfix(dict(loss=f"{loss.item():.3f}"))

    total = 0
    correct = 0
    for x, y in test_loader:
        pred = model(x)
        total += len(x)
        # compare the last dim
        correct += (torch.argmax(pred, -1) == y).sum()

    print(f"{correct / total:.3f}")

# %%

for x, y in test_dataset[:20]:
    pred = model(torch.LongTensor([x]))[0]
    pred = torch.argmax(pred, -1).item()

    print(f"{id_to_word[x[2]]} pred:{id_to_pos[pred]} true:{id_to_pos[y]}")

# %%
