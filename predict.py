

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import os, sys

LABELS = [
    'INTJ','INTP','ENTJ','ENTP',
    'INFJ','INFP','ENFJ','ENFP',
    'ISTJ','ISFJ','ESTJ','ESFJ',
    'ISTP','ISFP','ESTP','ESFP'
]
NUM_CLASSES = len(LABELS)
MAX_LEN     = 256
MSDROPS     = 4

# ────────────────────────────────────────────────────────────────
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
    def __init__(self, hid, kernels=(3,4,5), filters=128, drop=0.5):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(hid, filters, k) for k in kernels])
        self.spatial = SpatialSE(filters * len(kernels))
        self.drop = drop
        self.out_dim = filters * len(kernels) + 3*hid
    def forward(self, seq):
        glob = torch.cat([seq[:,0], seq.mean(1), seq.max(1).values], 1)
        x = seq.transpose(1,2)
        feats = [F.relu(c(x)).max(-1).values for c in self.convs]
        cat = self.spatial(torch.cat(feats,1))
        return F.dropout(torch.cat([cat, glob],1), self.drop, self.training)

class MCMSDropClassifier(nn.Module):
    def __init__(self, in_dim, n=MSDROPS, p=0.5, num_classes=NUM_CLASSES):
        super().__init__(); self.n,self.p = n,p
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        outs = [self.fc(F.dropout(x, self.p, self.training)) for _ in range(self.n)]
        return torch.stack(outs).mean(0)

class MBTIModel(nn.Module):
    def __init__(self, base='FacebookAI/xlm-roberta-base',
                 kernels=(3,4,5), filters=128, heads=4):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(base)
        hid = self.encoder.config.hidden_size
        self.tattn   = TemporalAttn(hid, heads)
        self.textcnn = TextCNNBlock(hid, kernels, filters)
        self.cls_head = MCMSDropClassifier(self.textcnn.out_dim)
    def forward(self, ids, mask):
        seq = self.encoder(ids, attention_mask=mask).last_hidden_state
        seq = self.tattn(seq, mask)
        feat = self.textcnn(seq)
        return self.cls_head(feat)

# ────────────────────────────────────────────────────────────────
def load_model(model_dir='final_xlmr_mbti16_ta_cnn'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok = XLMRobertaTokenizer.from_pretrained(model_dir)
    net = MBTIModel().to(device)
    ckpt = torch.load(os.path.join(model_dir, 'best_model.pt'),
                      map_location=device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    return tok, net, device

@torch.no_grad()
def infer(text, tok, net, device):
    enc = tok(text, padding='max_length', truncation=True,
              max_length=MAX_LEN, return_tensors='pt')
    ids = enc['input_ids'].to(device)
    msk = enc['attention_mask'].to(device)
    prob = F.softmax(net(ids, msk), -1).cpu().squeeze()   # (16,)
    idx  = prob.argmax().item()
    return LABELS[idx], prob[idx].item()

# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model_dir = 'final_xlmr_mbti16_ta_cnn'
    if not os.path.isdir(model_dir):
        sys.exit(f"❌ 找不到模型目录 {model_dir}")
    print("⏳ 正在加载模型，请稍候…")
    tokenizer, model, device = load_model(model_dir)
    print("✅ 模型加载完毕！直接输入一句话，回车即可预测 MBTI；输入 exit/quit 结束\n")

    while True:
        try:
            text = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！"); break
        if text.lower() in {"exit", "quit"} or text == "":
            print("再见！"); break
        mbti, p = infer(text, tokenizer, model, device)
        print(f"🧠  预测类型: {mbti}")
