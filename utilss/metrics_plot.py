import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
# SB3_6Callback_vis/Plotpy_Matplot_评估/m24_保存到本地OK.py
# 1.系列指标
def calculate_metrics_series(history_dict, risk_free_rate=0.0):
    """
        # 折线图数据格式
        {
            "日收益率": {
                "series": [pd.Series1, pd.Series2, ...],  # 列表，而非dict
                "table": [
                    {"model": "PPO1_MLP", ...},
                    {"model": "PPO2_Lstm", ...}
                ]
            },
            ...
        }

        # 汇总表格数据（同上面的）
        [
        {'model': 'PPO1_MLP', 'max': 0.0099, 'min': 0.0, 'mean': 0.0089}
        {'model': 'PPO2_Lstm', 'max': 0.0099, 'min': 0.0, 'mean': 0.0089}
        ]

    """
    data_dict = {}

    for policy_name, history in history_dict.items():
        # net_worth = np.array(history['total_net_worth'])
        # profit = np.array(history['total_profit'])
        # rewards = np.array(history['total_reward'])
        # # dates = pd.date_range(end=history['current_data'], periods=len(net_worth))
        dates = pd.to_datetime(history['current_data'])
        # index = pd.to_datetime(historypre['current_data'])


        # 1. 计算每日收益率
        # 计算每日收益率,系列，
        # np.diff(net_worth)：计算净值序列的相邻差值，net_worth[:-1]：取前n-1日的净值作为分母
        # 该计算方式与pandas的pct_change()方法等价：
        # daily_returns = np.diff(net_worth) / net_worth[:-1]

      
        # 假设 history["total_net_worth"] 已经有数据
        net_worth_series = pd.Series(history['total_net_worth'],dates)
        # 计算每日收益率
        daily_returns = net_worth_series.pct_change().fillna(0)  # 第一天收益率设为0

        # 2. 波动率

       

        # Step 2: 逐步计算从 day 1 到 t 的累积标准差
        #daily_returns.expanding()：创建扩展窗口对象，第1个窗口：[0]，第2个窗口：[0,1]，第3个窗口：[0,1,2]
        daily_volatility = daily_returns.expanding().std().fillna(0)

         # # 计算从开始时间到当前步的波动率，和上面等价
        # volatility = [np.std(daily_returns[:i+1]) for i in range(len(daily_returns))]
    
        # 计算 5 日滚动标准差作为波动率
        # daily_volatility = daily_returns.rolling(window=5).std().fillna(0)


     

        # 添加到每个指标的 data_dict
        def add_to_data_dict(name, series_data,policy_name, table_value=None):
            if name not in data_dict:
                data_dict[name] = {"series": [], "table": []}
            data_dict[name]["series"].append(
                pd.Series(series_data, index=dates, name=policy_name)
            )
            data_dict[name]["table"].append(
                {"model": policy_name, "max": np.max(series_data),
                 "min": np.min(series_data), "mean": np.mean(series_data) if hasattr(series_data, "__len__") else series_data}
            )
         # 原始数据
        # 构造 pandas.Series，索引为日期 # x = pd.to_datetime(historypre['current_data'])
        series_net = pd.Series(history['total_net_worth'], index=dates, name=policy_name)
        series_profit = pd.Series(history['total_profit'], index=dates, name=policy_name)
        series_reward = pd.Series(history['total_reward'], index=dates, name=policy_name)

        for metric_name, series in zip(
            ["total_net_worth", "total_profit", "total_reward"],
            [series_net, series_profit, series_reward]
        ):
            add_to_data_dict(metric_name, series, policy_name,table_value=None)

        # 计算的指标数据
        add_to_data_dict("日收益率", daily_returns,policy_name, table_value=None)
        add_to_data_dict("日波动率",daily_volatility,policy_name, table_value=None)
        
    return data_dict



# 根据评估指标，规范化代码

def calculate_metrics_values(history_dict, risk_free_rate=0.0):
    data_list = []

    for policy_name, history in history_dict.items():
        # 数据提取与转换
        net_worth = np.array(history['total_net_worth'])
        # profit = np.array(history['total_profit']) # 此处未使用，注释掉
        # rewards = np.array(history['total_reward']) # 此处未使用，注释掉
        dates = pd.to_datetime(history['current_data'])

        # 构建时间序列
        net_worth_series = pd.Series(net_worth, index=dates)
        
        # 初始价值 P0 和 期末价值 Pend
        P_0 = net_worth[0]
        P_end = net_worth[-1]

        # ---------------------------------------------------------------------
        # 1. 累计收益率 (Cumulative Return, CR)
        # 公式: (Pend - P0) / P0
        # ---------------------------------------------------------------------
        total_return = (P_end - P_0) / P_0

        # ---------------------------------------------------------------------
        # 2. 最大收益率 (Maximum Earning Rate, MER)
        # 公式: (max(A) - A0) / A0
        # ---------------------------------------------------------------------
        max_val = net_worth.max()
        max_return = (max_val - P_0) / P_0

        # ---------------------------------------------------------------------
        # 3. 最大回撤 (Maximum Drawdown, MDD)
        # 公式: max( (RollMax - Current) / RollMax )
        # 注意：公式定义的 MDD 为正数（损失幅度的绝对值）
        # ---------------------------------------------------------------------
        roll_max = net_worth_series.cummax()
        # 计算回撤序列 (正值代表回撤幅度)
        drawdown_series = (roll_max - net_worth_series) / roll_max 
        max_drawdown = drawdown_series.max()

        # ---------------------------------------------------------------------
        # 4. 每笔平均盈利能力 (APPT)
        # 公式: (Pend - P0) / NT
        # ---------------------------------------------------------------------
        total_trades_list = np.array(history['total_trades'])
        num_trades = total_trades_list[-1] if len(total_trades_list) > 0 else 0
        
        # 防止除以零
        appt = (P_end - P_0) / num_trades if num_trades > 0 else 0.0

        # ---------------------------------------------------------------------
        # 基础收益率序列计算 (用于波动率、夏普、均值等)
        # ---------------------------------------------------------------------
        # 日收益率 R_t
        daily_returns = net_worth_series.pct_change().dropna()

        # ---------------------------------------------------------------------
        # 5. 波动率 (Volatility, VOL)
        # 公式: std(R) * sqrt(252)
        # ---------------------------------------------------------------------
        volatility = daily_returns.std() * np.sqrt(252)

        # ---------------------------------------------------------------------
        # 6. 夏普比率 (Sharpe Ratio, SR)
        # 公式: (mean(R) - Rf) / std(R) * sqrt(252)
        # ---------------------------------------------------------------------
        std_dev = daily_returns.std()
        if std_dev != 0:
            sharpe_ratio = ((daily_returns.mean() - risk_free_rate) / std_dev) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # ---------------------------------------------------------------------
        # 7. 卡玛比率 (Calmar Ratio, CR)
        # 公式: Total Return / |MDD|
        # ---------------------------------------------------------------------
        # MDD 已经是正数，直接作为分母
        calmar_ratio = total_return / max_drawdown if max_drawdown != 0 else np.inf

        # ---------------------------------------------------------------------
        # 8-11. 各周期均值收益率 (DAR, MAR, QAR, AAR)
        # 公式: mean(Periodic Returns)
        # ---------------------------------------------------------------------
        # 日均
        daily_avg_return = daily_returns.mean()
        
        # 月均 (基于月末价值百分比变化)
        monthly_returns = net_worth_series.resample('ME').last().pct_change().dropna()
        monthly_avg_return = monthly_returns.mean()

        # 季度均 (基于季末价值百分比变化)
        quarterly_returns = net_worth_series.resample('QE').last().pct_change().dropna()
        quarterly_avg_return = quarterly_returns.mean()

        # 年均 (基于年末价值百分比变化)
        annual_returns = net_worth_series.resample('YE').last().pct_change().dropna()
        annual_avg_return = annual_returns.mean() if not annual_returns.empty else np.nan

        # ---------------------------------------------------------------------
        # 存入列表
        # ---------------------------------------------------------------------
        data_list.append({
            "model": policy_name,
            # 主要指标
            # "累计收益率": total_return,      # CR  
            "收益率": total_return,      # CR  为了兼容之前版本，将累计收益率改成收益率
            "最大收益率": max_return,        # MER
            "最大回撤": max_drawdown,        # MDD (正数)
            "APPT": appt,                   # APPT
            "波动率": volatility,            # VOL
            "夏普比率": sharpe_ratio,        # SR
            "卡玛比率": calmar_ratio,        # CR (Calmar)
            
            # 辅助信息
            "累计净值": P_end,
            
            # 周期性均值指标
            "日均收益率": daily_avg_return,      # DAR
            "月均收益率": monthly_avg_return,    # MAR
            "季度均收益率": quarterly_avg_return,# QAR
            "年均收益率": annual_avg_return,     # AAR
        })

    return data_list




# 3.绘图
def plot_metrics_and_summary(data_dict, summary_values,base_path="./charts",show=True):
     # 3、绘图
    # history_dict={
    #     'PPO1_MLP': history,
    #     'PPO2_Lstm': history,
    # }
    # # data_dict=metrics(history_dict)
    # data_dict = calculate_metrics_series(history_dict)  # 生成时序和指标表格
    # summary_values = calculate_metrics_values(history_dict)  # 生成整体指标汇总

    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots


    # 总共有多少组指标,（折线图 + 表格）+(表格)
    n_metrics = len(data_dict) 
    n_table = 1 #len(summary_values)

    # 创建子图：每组两行（折线图 + 表格），+(表格)
    fig = make_subplots(
        rows=n_metrics * 2+n_table, 
        cols=1,
        row_heights=[0.6, 0.4] * n_metrics+[0.5]*n_table,
        # vertical_spacing=0.1,
        vertical_spacing=0.02,  # 间距改小
        specs=[[{"type": "xy"}], [{"type": "table"}]] * n_metrics+ [[{"type": "table"}]] * n_table,
        subplot_titles=[
            f"{metric} 折线图" if i % 2 == 0 else f"{metric} 表格"
            for metric in data_dict for i in range(2)
        ]+[f"汇总指标表格{i}" for i in range(n_table)],
    )

    row = 1
    for metric_name, metric_data in data_dict.items():
        # 折线图部分
        for s in metric_data["series"]:
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name=f"{metric_name} - {s.name}"
                ),
                row=row, col=1
            )
        
        # 表格部分
        table = metric_data["table"]
        if not table:
            print(f"[警告] 表格为空: {metric_name}")
        # else:
        #     print(f"[信息] 表格不为空: {table}")
        #     for item in table:
        #         print(f"[信息] 表格内容: {item}")
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["模型", "最大值", "最小值", "平均值"],
                    fill_color="lightblue",
                    align="left"
                ),
                cells=dict(
                    # values=[
                    #     [item["model"] for item in table],
                    #     [item["max"] for item in table],
                    #     [item["min"] for item in table],
                    #     [item["mean"] for item in table],
                    # ],
                    values=[
                        [item["model"] for item in table],
                        [float(item["max"]) for item in table],
                        [float(item["min"]) for item in table],
                        [float(item["mean"]) for item in table],
                    ],
                    fill_color="lavender",
                    align="left"
                )
            ),
            row=row + 1, col=1
        )
        row += 2

        #  # 表格
        # table_df = pd.DataFrame(metric_data["table"])
        # fig.add_trace(
        #     go.Table(
        #         header=dict(values=list(table_df.columns), fill_color="lightgrey", align="left"),
        #         cells=dict(values=[table_df[col] for col in table_df.columns], fill_color="white", align="left")
        #     ),
        #     row=row + 1, col=1
        # )
        # row += 2

    # 汇总表格部分
    summary_df = pd.DataFrame(summary_values)
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(summary_df.columns),
                fill_color="lightblue",
                align="left"
            ),
            cells=dict(
                values=[summary_df[col] for col in summary_df.columns],
                fill_color="lavender",
                align="left"
            )
        ),
        row=row, col=1
    )
    
       

    # 布局调整
    fig.update_layout(
        height=600 * n_metrics,
        width=1000,
        title="多组指标：折线图 + 表格（含假数据）",
        showlegend=True,
        # margin=dict(t=40),  # 顶部边距，避免标题太靠上
        margin=dict(t=40, b=40, l=40, r=40),  # 上下左右边距
        autosize=True,  # 自动调整大小
    )
    if show:
        fig.show()

    # 显示图表
    # fig.show()

   #保存到本地
   
    # 保存图表
    # 创建目录
    os.makedirs("charts", exist_ok=True)

    # 当前时间
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    filename_str = now.strftime("%Y-%m-%d_%H-%M-%S")

    # filepath = f"charts/linechart_{filename_str}.html" #base_path
    filepath = f"{base_path}/linechart_{filename_str}.html" #base_path
    fig.write_html(filepath)
    print(f"✅ 图表保存：{filepath}")




import matplotlib.pyplot as plt



# 3.绘图 Matplotlib，双 y 轴
def matplot_show(historypre, base_path="./charts2", show=True, data=True):
    """
    绘制 Total Net Worth / Total Profit / Total Reward 曲线
    若 Total Reward 数值远小于其他指标，则自动使用双 y 轴显示 Reward
    """

    # X轴数据
    if data:
        x = pd.to_datetime(historypre['current_data'])
    else:
        x = np.arange(len(historypre['current_data']))

    # Y轴数据
    y1 = np.array(historypre['total_net_worth'])
    y2 = np.array(historypre['total_profit'])
    y3 = np.array(historypre['total_reward'])

    # 判断是否需要双y轴
    use_dual_axis = (max(y1.max(), y2.max()) / (y3.max() + 1e-8)) > 10  # 差距超过10倍就启用双轴

    if use_dual_axis:
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # 左轴 Net Worth / Profit
        ax1.plot(x, y1, label='Total Net Worth', color='blue')
        ax1.plot(x, y2, label='Total Profit', color='orange')
        ax1.set_xlabel('Current Data')
        ax1.set_ylabel('Value')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        # 右轴 Reward
        ax2 = ax1.twinx()
        ax2.plot(x, y3, label='Total Reward', color='green')
        ax2.set_ylabel('Reward')
        ax2.legend(loc='upper right')

        plt.title('Metrics over Current Data (Dual Axis)')
    else:
        plt.figure(figsize=(12, 6))
        plt.plot(x, y1, label='Total Net Worth')
        plt.plot(x, y2, label='Total Profit')
        plt.plot(x, y3, label='Total Reward', color='green')
        plt.xlabel('Current Data')
        plt.ylabel('Value')
        plt.title('Metrics over Current Data')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    # 显示图像
    if show:
        plt.show()

    # 保存图像
    os.makedirs(base_path, exist_ok=True)
    filename_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filepath = f"{base_path}/Matfig_{filename_str}.jpg"
    plt.savefig(filepath)
    print(f"✅ Mat图片保存：{filepath}")

    plt.close()



if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # 随机种子
    np.random.seed(42)

    n_steps = 100
    initial_net_worth = 10000
    price_changes = np.random.normal(loc=0.001, scale=0.01, size=n_steps)

    # 系列数据：净值、利润、奖励
    net_worth_series = initial_net_worth * np.cumprod(1 + price_changes)  # ✅ 系列
    profit_series = net_worth_series - initial_net_worth                  # ✅ 系列
    reward_series = price_changes * 100                                   # ✅ 系列

    history = {
        'total_net_worth': net_worth_series.tolist(),
        'total_profit': profit_series.tolist(),
        'total_reward': reward_series.tolist(),
        # 'total_trades': [0] * len(net_worth_series),
        'total_trades': [np.random.randint(1, 10) for _ in range(len(net_worth_series))],  # 假设每个时间步有1-10笔交易
        'current_data': pd.date_range('2023-01-01', periods=len(net_worth_series), freq='D').strftime('%Y-%m-%d').tolist()  # 存储格式化后的日期列表
    }

    # print(history)

    print("\n历史数据最后5条记录:")
    for key in history:
        print(f"{key}:")
        # 对数值数据保留4位小数，日期保持原格式
        values = [round(x, 4) if isinstance(x, (float, np.floating)) else x 
                for x in history[key][-5:]]  # 可修改-5为其他数值调整显示数量
        print(values)

    matplot_show(history,base_path="./charts")

   
# #### ************ceshi************
    # 1、计算指标，系列
    data_dict1 = calculate_metrics_series({'PPO1_MLP': history, 'PPO2_Lstm':history})
    print("\n方法一：评估指标:")

    # 新增打印代码
    print("各指标前3条数据预览：")
    for metric_name in data_dict1:
        print(f"\n=== {metric_name} ===")
        
        # 打印时序数据前3条
        print(f"时序数据（{len(data_dict1[metric_name]['series'])}个模型）:")
        for series in data_dict1[metric_name]['series']:
            print(f"{series.name}: {[round(x,4) for x in series.values[:3]]}...")
        
        # 打印表格数据前3条
        print(f"\n表格统计（前3项）:")
        for table_item in data_dict1[metric_name]['table'][:3]:
            print({k: round(v,4) if isinstance(v, float) else v 
                for k, v in table_item.items()})
            
     # 2、计算评估指标值，多值
    data_dict2 = calculate_metrics_values({'PPO1_MLP': history, 'PPO2_Lstm':history})
    print("\n方法二：评估指标:")
    print(data_dict2)

    

    

    # 3、绘图
    history_dict={
        'PPO1_MLP': history,
        'PPO2_Lstm': history,
    }
    # data_dict=metrics(history_dict)
    data_dict = calculate_metrics_series(history_dict)  # 生成时序和指标表格
    summary_values = calculate_metrics_values(history_dict)  # 生成整体指标汇总

    plot_metrics_and_summary(data_dict, summary_values,base_path="./charts")


