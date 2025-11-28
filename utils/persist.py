import pandas as pd


import pandas as pd
import os

def persist_learning_history(history, name):
    # 检查文件是否存在且不为空
    if os.path.exists("learning_history.csv") and os.path.getsize("learning_history.csv") > 0:
        try:
            hist = pd.read_csv("learning_history.csv")
        except pd.errors.EmptyDataError:
            # 如果文件存在但为空，创建新的DataFrame
            hist = pd.DataFrame()
    else:
        # 如果文件不存在，创建新的DataFrame
        hist = pd.DataFrame()

    # 确保history是列表类型
    if not isinstance(history, list):
        history = [history]

    # 新增一列
    hist[name] = history
    # 保存回文件
    hist.to_csv("learning_history.csv", index=False)


if __name__ == '__main__':
    loss_history = [0,1,2,3,4,5,6,7,8,9]
    persist_learning_history(loss_history, "distill_loss")

