# ════════════════════════════════════════════════════════════════════
# 1. 依赖
# ════════════════════════════════════════════════════════════════════
from sklearn.metrics import classification_report, f1_score, roc_curve, auc
import os, random, math, json, time, gc, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import (XLMRobertaTokenizer, XLMRobertaModel,
                          AdamW, get_linear_schedule_with_warmup)

from tqdm import tqdm
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ════════════════════════════════════════════════════════════════════
# 2. 全局设置
# ════════════════════════════════════════════════════════════════════
SEED          = 42
BATCH_SIZE    = 16
GRAD_ACCUM    = 2          # 累计 2×16 = 32 实际 batch
EPOCHS        = 10
PATIENCE      = 2          # Early-Stopping
MAX_LEN       = 256
LR_HEAD       = 5e-5
LR_ENCODER    = 2e-5
LR_DECAY      = 0.95       # LLRD
WARMUP_RATIO  = 0.2
DROPOUT_P     = 0.5
MSDROPS       = 4
LABELS        = [
    'INTJ','INTP','ENTJ','ENTP',
    'INFJ','INFP','ENFJ','ENFP',
    'ISTJ','ISFJ','ESTJ','ESFJ',
    'ISTP','ISFP','ESTP','ESFP'
]
NUM_CLASSES   = len(LABELS)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ════════════════════════════════════════════════════════════════════
# 3. 数据加载（分层抽样）
# ════════════════════════════════════════════════════════════════════
df = (pd.read_csv('qna_cleaned.tsv', sep='\t',
                  usecols=['answer','a_mbti'])
        .dropna(subset=['answer','a_mbti']))

df['a_mbti'] = df['a_mbti'].str.strip().str.lower()   
df['a_mbti'] = df['a_mbti'].str.upper()               


# 将 16-类字符串标签映射为整数
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}
df = df[df['a_mbti'].isin(LABELS)]          
df['label'] = df['a_mbti'].map(label2id)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED,
                                    stratify=df['label'])
train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════
# 4. Dataset
# ════════════════════════════════════════════════════════════════════
class MBTIDataset(Dataset):
    def __init__(self, df, tok, max_len):
        self.df, self.tok, self.max_len = df, tok, max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        text  = str(self.df.loc[idx, 'answer']).replace("[SEP]", " ")  
        label = torch.tensor(self.df.loc[idx, 'label'], dtype=torch.long)
        enc = self.tok(text, padding='max_length', truncation=True,
                       max_length=self.max_len, return_tensors='pt')
        return enc['input_ids'][0], enc['attention_mask'][0], label

def collate(batch):
    ids, msk, y = zip(*batch)
    return torch.stack(ids), torch.stack(msk), torch.tensor(y)

# ════════════════════════════════════════════════════════════════════
# 5. 模型组件
# ════════════════════════════════════════════════════════════════════
class TemporalAttn(nn.Module):
    def __init__(self, dim, heads=4, drop=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, drop, batch_first=True)
        self.ln   = nn.LayerNorm(dim)
    def forward(self, x, mask):
        out,_ = self.attn(x, x, x, key_padding_mask=~mask.bool())
        return self.ln(x + out)

class SpatialSE(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(ch, ch//r), nn.Linear(ch//r, ch)
    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x))))
        return x * w

class TextCNNBlock(nn.Module):
    def __init__(self, hid, kernels=(3,4,5), filters=128, drop=DROPOUT_P):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid, filters, k) for k in kernels])
        self.spatial = SpatialSE(filters * len(kernels))
        self.drop = drop
        self.out_dim = filters * len(kernels) + 3*hid
    def forward(self, seq):                     # B,T,H
        glob = torch.cat([seq[:,0], seq.mean(1), seq.max(1).values], 1)  # B,3H
        x = seq.transpose(1,2)                  # B,H,T
        feats = [F.relu(c(x)).max(-1).values for c in self.convs]  # each B,C
        cat = torch.cat(feats,1)                # B,C
        cat = self.spatial(cat)
        return F.dropout(torch.cat([cat, glob],1), self.drop, self.training)

class MCMSDropClassifier(nn.Module):

    def __init__(self, in_dim, n=MSDROPS, p=DROPOUT_P, num_classes=NUM_CLASSES):
        super().__init__(); self.n,self.p,self.num_classes = n,p,num_classes
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        outs = [self.fc(F.dropout(x, self.p, self.training)) for _ in range(self.n)]
        return torch.stack(outs).mean(0)

# ════════════════════════════════════════════════════════════════════
# 6. 整体模型
# ════════════════════════════════════════════════════════════════════
class MBTIModel(nn.Module):
    def __init__(self, base='FacebookAI/xlm-roberta-base',
                 kernels=(3,4,5), filters=128, heads=4):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(base)
        self.encoder.gradient_checkpointing_enable()  # 显存友好
        hid = self.encoder.config.hidden_size
        self.tattn   = TemporalAttn(hid, heads)
        self.textcnn = TextCNNBlock(hid, kernels, filters)
        dim = self.textcnn.out_dim
        self.cls_head = MCMSDropClassifier(dim, num_classes=NUM_CLASSES)
    def forward(self, ids, mask):
        seq = self.encoder(ids, attention_mask=mask).last_hidden_state
        seq = self.tattn(seq, mask)
        feat = self.textcnn(seq)
        return self.cls_head(feat)              # B,16

# ════════════════════════════════════════════════════════════════════
# 7. 初始化
# ════════════════════════════════════════════════════════════════════
tok   = XLMRobertaTokenizer.from_pretrained('FacebookAI/xlm-roberta-base')
model = MBTIModel().to(device)

train_dl = DataLoader(MBTIDataset(train_df,tok,MAX_LEN),
                      BATCH_SIZE, True,  collate_fn=collate)
val_dl   = DataLoader(MBTIDataset(val_df,tok,MAX_LEN),
                      BATCH_SIZE, False, collate_fn=collate)

# 类别权重（防止类别不平衡）
cls_cnts = train_df['label'].value_counts().sort_index().values
cls_weights = torch.tensor(1/cls_cnts, dtype=torch.float, device=device)
cls_weights = cls_weights / cls_weights.sum() * NUM_CLASSES   # 归一化

crit = nn.CrossEntropyLoss(weight=cls_weights)
scaler = torch.cuda.amp.GradScaler()

# LLRD param groups
layers = ([model.encoder.embeddings] +
          list(model.encoder.encoder.layer))
pg = []
for i,layer in enumerate(layers[::-1]):   # 顶→底
    pg.append({'params': layer.parameters(),
               'lr': LR_ENCODER * (LR_DECAY**i)})
pg += [{'params':model.tattn.parameters(),      'lr':LR_HEAD},
       {'params':model.textcnn.parameters(),    'lr':LR_HEAD},
       {'params':model.cls_head.parameters(),   'lr':LR_HEAD}]
opt = AdamW(pg, weight_decay=1e-3)

steps = math.ceil(len(train_dl)/GRAD_ACCUM)*EPOCHS
sched = get_linear_schedule_with_warmup(opt,
        int(WARMUP_RATIO*steps), steps)

# ════════════════════════════════════════════════════════════════════
# 8. 评估
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(net, dl):
    net.eval(); ys, ps, losses = [],[],[]
    for ids,msk,y in dl:
        ids,msk,y = ids.to(device),msk.to(device),y.to(device)
        with torch.cuda.amp.autocast():
            logit = net(ids,msk); loss = crit(logit,y)
        losses.append(loss.item())
        ys.append(y.cpu()); ps.append(F.softmax(logit,1).cpu())
    y = torch.cat(ys).numpy()          # (N,)
    p = torch.cat(ps).numpy()          # (N,16)

    # 预测标签
    preds = p.argmax(1)
    macro_f1 = f1_score(y, preds, average='macro')

    print("\n── 16 类 MBTI 分类报告（验证集）──\n",
          classification_report(y, preds, target_names=LABELS,
                                digits=4, zero_division=0))

    # 按一对多绘制 ROC 用的二值化指标
    y_bin = np.eye(NUM_CLASSES)[y]
    return np.mean(losses), macro_f1, y_bin, p

# ════════════════════════════════════════════════════════════════════
# 9. 训练
# ════════════════════════════════════════════════════════════════════
best_f1, patience = 0, 0
f1_hist = []

out_dir = 'final_xlmr_mbti16_ta_cnn'
os.makedirs(out_dir, exist_ok=True)

for ep in range(1,EPOCHS+1):
    model.train(); tot=0; opt.zero_grad()
    pbar = tqdm(enumerate(train_dl), total=len(train_dl),
                desc=f"Epoch {ep}/{EPOCHS}")
    for step,(ids,msk,y) in pbar:
        ids,msk,y = ids.to(device),msk.to(device),y.to(device)
        with torch.cuda.amp.autocast():
            logit = model(ids,msk); loss = crit(logit,y)/GRAD_ACCUM
        scaler.scale(loss).backward()
        if (step+1)%GRAD_ACCUM==0 or (step+1)==len(train_dl):
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
        tot += loss.item()*GRAD_ACCUM
    train_loss = tot/len(train_dl)

    val_loss,macro_f1, y_val, p_val = evaluate(model,val_dl)
    f1_hist.append(macro_f1)

    print(f"\nEpoch {ep} | Train {train_loss:.4f}  Val {val_loss:.4f} "
          f"Macro-F1 {macro_f1:.4f}")

    if macro_f1 > best_f1 + 1e-4:
        best_f1, patience = macro_f1, 0
        torch.save({'model':model.state_dict()}, f'{out_dir}/best_model.pt')
        print("  ✅  New best; model saved.")
    else:
        patience += 1
        if patience > PATIENCE:
            print("  ⏹️  Early stopping triggered.")
            break

# ════════════════════════════════════════════════════════════════════
# 10. 结果绘制
# ════════════════════════════════════════════════════════════════════
plt.figure(figsize=(7,5))
plt.plot(range(1,len(f1_hist)+1), f1_hist, marker='o')
plt.xlabel('Epoch'); plt.ylabel('Macro-F1')
plt.title('Validation Macro-F1 Curve'); plt.grid(); plt.tight_layout()
plt.savefig(f'{out_dir}/f1_curve.png')

plt.figure(figsize=(7,7))
for i,n in enumerate(LABELS):
    fpr,tpr,_ = roc_curve(y_val[:,i], p_val[:,i]); roc = auc(fpr,tpr)
    plt.plot(fpr,tpr,label=f"{n} (AUC={roc:.4f})")
plt.plot([0,1],[0,1],'--',lw=0.8); plt.xlabel('FPR'); plt.ylabel('TPR')
plt.title('One-vs-Rest ROC Curves (Validation)'); plt.legend(fontsize=8)
plt.grid(); plt.tight_layout()
plt.savefig(f'{out_dir}/roc_auc.png')

# ════════════════════════════════════════════════════════════════════
# 11. 导出
# ════════════════════════════════════════════════════════════════════
tok.save_pretrained(out_dir)
print(f"\n🎉  训练完毕，最优 Macro-F1 {best_f1:.4f}")
print(f"✅  模型已保存到  {out_dir}")
print("📈  曲线已保存为  f1_curve.png  /  roc_auc.png")
