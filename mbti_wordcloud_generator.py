
import pandas as pd
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from konlpy.tag import Okt
import os

# ════════════════════════════════════════════════════════════════════
# 1. 设置
# ════════════════════════════════════════════════════════════════════
FONT_PATH = "NanumGothicCoding.ttf"  # 韩文字体路径
DATA_FILE = "qna_cleaned.tsv"
OUTPUT_DIR = "wordcloud_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABELS = [
    'INTJ','INTP','ENTJ','ENTP',
    'INFJ','INFP','ENFJ','ENFP',
    'ISTJ','ISFJ','ESTJ','ESFJ',
    'ISTP','ISFP','ESTP','ESFP'
]

okt = Okt()

# ════════════════════════════════════════════════════════════════════
# 2. 数据加载 
# ════════════════════════════════════════════════════════════════════
df = pd.read_csv(DATA_FILE, sep='\t', usecols=['answer', 'a_mbti'])
df = df.dropna(subset=['answer', 'a_mbti'])

# 标签转大写
df['a_mbti'] = df['a_mbti'].str.upper()

# 去除 [SEP] 等标记（方便分词）
df['answer'] = df['answer'].str.replace(r'\[SEP\]', ' ', regex=True)

# 保证标签合法
df = df[df['a_mbti'].isin(LABELS)]

print("\n📌 数据集前5行内容：")
print(df.head(5))

# 类别统计
print("\n🟢 每个类别的样本数量：")
print(df['a_mbti'].value_counts())

# ════════════════════════════════════════════════════════════════════
# 3. 提取词频函数
# ════════════════════════════════════════════════════════════════════
def get_word_freq(texts):
    words = []
    for text in texts:
        tokens = okt.morphs(str(text))  # 使用所有分词（更全面）
        words.extend([w for w in tokens if len(w) > 1])  # 过滤单字、标点
    return Counter(words)

# ════════════════════════════════════════════════════════════════════
# 4. 绘制词云函数
# ════════════════════════════════════════════════════════════════════
def plot_wordcloud(word_freq, title, save_path):
    if not word_freq:  # 防止空的情况
        print(f"⚠️  {title} 没有足够的词，跳过绘图。")
        return
    wc = WordCloud(font_path=FONT_PATH,
                   background_color='white',
                   width=800, height=600,
                   max_words=100).generate_from_frequencies(word_freq)
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ════════════════════════════════════════════════════════════════════
# 5. 绘制每个类别的词云
# ════════════════════════════════════════════════════════════════════
print("\n🎨 正在生成每个 MBTI 类别的词云...")
for label in LABELS:
    subset = df[df['a_mbti'] == label]['answer']
    freq = get_word_freq(subset)
    plot_wordcloud(freq, f"MBTI Type: {label}",
                   f"{OUTPUT_DIR}/wordcloud_{label}.png")
    print(f"✅  {label} 词云已生成")

# ════════════════════════════════════════════════════════════════════
# 6. 绘制总的词云
# ════════════════════════════════════════════════════════════════════
print("\n🎉 正在生成全体总词云...")
all_freq = get_word_freq(df['answer'])
plot_wordcloud(all_freq, "Total Wordcloud",
               f"{OUTPUT_DIR}/wordcloud_total.png")
print("✅  总词云已生成")

print(f"\n📂  词云图片已保存在：{OUTPUT_DIR}/")
