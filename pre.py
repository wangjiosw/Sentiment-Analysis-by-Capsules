import codecs
import pandas as pd


if __name__ == "__main__":
    data_neg = codecs.open('data/rt-polarity.neg', 'r', encoding='utf-8', errors='ignore').read().split('\n')
    data_pos = codecs.open('data/rt-polarity.pos', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    data_neg = data_neg[:-1]
    data_pos = data_pos[:-1]

    # print(data_neg[:5],data_pos[:5])

    num = len(data_neg)

    # 0-neg, 1-pos
    label1 = [0]*num
    label2 = [1]*num

    data_neg.extend(data_pos)
    label1.extend(label2)

    raw_data = {'INPUT': data_neg, 'OUTPUT': label1}
    df = pd.DataFrame(raw_data, columns=["INPUT", "OUTPUT"])

    from sklearn.model_selection import train_test_split
    # create train and validation set
    train, val = train_test_split(df, test_size=0.1,random_state=1)
    train, test = train_test_split(train, test_size=0.2,random_state=1)

    train.to_csv("data/train.csv", index=False)
    val.to_csv("data/val.csv", index=False)
    test.to_csv("data/test.csv", index=False)
