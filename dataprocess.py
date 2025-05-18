import pandas as pd

# 读取原始数据
df = pd.read_csv('qna_cleaned.tsv', sep='\t')


df = df[['answer', 'a_mbti']].dropna()


def mbti_to_binary_labels(mbti):
    return pd.Series({
        'EI': 0 if mbti[0].upper() == 'E' else 1,  # E=0, I=1
        'NS': 0 if mbti[1].upper() == 'N' else 1,  # N=0, S=1
        'FT': 0 if mbti[2].upper() == 'F' else 1,  # F=0, T=1
        'PJ': 0 if mbti[3].upper() == 'P' else 1   # P=0, J=1
    })


labels_df = df['a_mbti'].apply(mbti_to_binary_labels)


output_df = pd.concat([df['answer'], df['a_mbti'], labels_df], axis=1)

# 保存到新的 CSV 文件
output_df.to_csv('qna_cleaned_binary_labels.csv', index=False)

print("success")
