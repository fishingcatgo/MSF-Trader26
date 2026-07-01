
import pandas as pd




# =========================
# 2️⃣ 切分函数（带窗口）
# =========================
def data_split_with_window(df, start, end, window_size=0, target_date_col="date"):
    """
    按日期切分数据集，并可在验证/交易集前加上前面 window_size 日期
    :param df: pandas DataFrame, 必须包含 target_date_col 和 'tic'
    :param start: 起始日期（字符串或 Timestamp）
    :param end: 结束日期（字符串或 Timestamp）
    :param window_size: int, 交易集或验证集前加上训练集最后的 window_size 日期
    :param target_date_col: 时间列名称
    :return: pandas DataFrame
    """
    # 主区间切分
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    
    # 如果需要窗口前置，取 start 前的 window_size 日期
    if window_size > 0:
        prev_dates = df[df[target_date_col] < start][target_date_col].drop_duplicates().sort_values().tail(window_size)
        prev_data = df[df[target_date_col].isin(prev_dates)]
        data = pd.concat([prev_data, data], ignore_index=True)
    
    # 按 date 和 tic 排序
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    
    # 重新索引
    data.index = data[target_date_col].factorize()[0]
    
    return data