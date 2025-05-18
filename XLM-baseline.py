import os, random, torch, argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, f1_score,
                             roc_curve, auc)
from sklearn.preprocessing import label_binarize         # ★ new
from torch.utils.data import Dataset, DataLoader
from transformers import (XLMRobertaTokenizer, XLMRobertaModel,
                          AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm
import torch.nn as nn

# ─────────────────── 全局设置 ─────────────────── #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────── 数据加载（16 分类） ─────────────────── #
df = pd.read_csv('qna_cleaned.tsv', sep='\t')            
df = df[['answer', 'a_mbti']].dropna()                  

MBTI_TYPES = sorted(df['a_mbti'].unique().tolist())     
type2id = {t: i for i, t in enumerate(MBTI_TYPES)}
id2type = {i: t for t, i in type2id.items()}

df['label'] = df['a_mbti'].map(type2id)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df['label']   
)
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

# ─────────────────── 数据集类 ─────────────────── #
class MBTIDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text  = str(self.df.loc[idx, 'answer']).replace("[SEP]", " ")  
        label = torch.tensor(self.dataframe.loc[idx, 'label'], dtype=torch.long)
        enc = self.tokenizer(text, max_length=self.max_len,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        return enc['input_ids'].squeeze(), enc['attention_mask'].squeeze(), label

def collate_fn(batch):
    ids, masks, labels = zip(*batch)
    return torch.stack(ids), torch.stack(masks), torch.tensor(labels)

# ─────────────────── 模型结构 ─────────────────── #
class MBTIClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder   = XLMRobertaModel.from_pretrained(model_name)
        hidden_size    = self.encoder.config.hidden_size
        self.classifier= nn.Linear(hidden_size, num_classes)      # ★ changed

    def forward(self, input_ids, attention_mask):
        out     = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = out.last_hidden_state[:, 0]           # [CLS]
        logits  = self.classifier(pooled)               # [B,16]
        return logits

# ─────────────────── 超参数 & DataLoader ─────────────────── #
model_name   = 'FacebookAI/xlm-roberta-base'
tokenizer    = XLMRobertaTokenizer.from_pretrained(model_name)
num_classes  = len(MBTI_TYPES)

model        = MBTIClassifier(model_name, num_classes).to(device)

max_len      = 256
batch_size   = 32
epochs       = 4
lr           = 2e-5
weight_decay = 1e-3

train_loader = DataLoader(MBTIDataset(train_df, tokenizer, max_len),
                          batch_size=batch_size, shuffle=True,
                          collate_fn=collate_fn)
val_loader   = DataLoader(MBTIDataset(val_df, tokenizer, max_len),
                          batch_size=batch_size, collate_fn=collate_fn)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
total_steps= epochs * len(train_loader)
scheduler  = get_linear_schedule_with_warmup(optimizer,
              num_warmup_steps=int(0.1 * total_steps),
              num_training_steps=total_steps)

criterion  = nn.CrossEntropyLoss()                     # ★ changed
scaler     = torch.cuda.amp.GradScaler()

# ─────────────────── 评估函数 ─────────────────── #
@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    all_labels, all_probs, losses = [], [], []

    for ids, masks, labels in loader:
        ids, masks, labels = ids.to(device), masks.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            logits = net(ids, masks)
            loss   = criterion(logits, labels)
        losses.append(loss.item())

        probs = torch.softmax(logits, dim=-1)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)        # shape [N]
    all_probs  = np.array(all_probs)         # shape [N,16]
    preds      = np.argmax(all_probs, axis=1)
    macro_f1   = f1_score(all_labels, preds, average='macro')

    print("\n── 分类报告 ──\n",
          classification_report(all_labels, preds,
                                target_names=MBTI_TYPES, digits=4))
    return np.mean(losses), macro_f1, all_labels, all_probs

# ─────────────────── 训练循环 ─────────────────── #
f1_history = []

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0

    for ids, masks, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        ids, masks, labels = ids.to(device), masks.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(ids, masks)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss, val_macro_f1, val_labels, val_probs = evaluate(model, val_loader)
    f1_history.append(val_macro_f1)

    print(f"Epoch {epoch}: "
          f"Train Loss = {avg_train_loss:.4f} | "
          f"Val Loss = {val_loss:.4f} | "
          f"Val Macro-F1 = {val_macro_f1:.4f}")

# ─────────────────── 绘制 Macro-F1 曲线 ─────────────────── #
plt.figure(figsize=(6, 4))
x = list(range(1, epochs + 1))
plt.plot(x, f1_history, marker='o')
plt.xticks(x); plt.xlabel('Epoch'); plt.ylabel('Macro-F1 Score')
plt.title('Validation Macro-F1 Curve'); plt.grid(True); plt.tight_layout()

out_dir = 'trained_xlmr_mbti16_amp'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, 'macro_f1_curve.png'))

# ─────────────────── 绘制 ROC-AUC 曲线（OvR） ─────────────────── #
plt.figure(figsize=(7, 7))
# binarize labels for ROC
y_true = label_binarize(val_labels, classes=list(range(num_classes)))
for i, name in enumerate(MBTI_TYPES):
    fpr, tpr, _ = roc_curve(y_true[:, i], val_probs[:, i])
    roc_auc     = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', linewidth=0.8)
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title('ROC Curves (Validation Set)')
plt.legend(fontsize=7, loc='lower right'); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'roc_auc.png'))

# ─────────────────── 保存模型 ─────────────────── #
torch.save(model.state_dict(), os.path.join(out_dir, 'pytorch_model.bin'))
tokenizer.save_pretrained(out_dir)

print(f"\n✅ 半精度 模型和分词器已保存到 {out_dir}")
print("📈 Macro-F1 曲线保存为 macro_f1_curve.png ，ROC-AUC 曲线保存为 roc_auc.png")
