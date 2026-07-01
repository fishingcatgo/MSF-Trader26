from __future__ import annotations  # 启用未来的注解特性

from typing import List  # 导入类型提示

import gymnasium as gym  # 导入gymnasium库，用于自定义强化学习环境
import matplotlib  # 导入matplotlib用于绘图
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块
import numpy as np  # 导入numpy用于数值计算
import pandas as pd  # 导入pandas用于数据处理
from gymnasium import spaces  # 导入spaces用于定义动作空间和观测空间
from gymnasium.utils import seeding  # 导入seeding用于环境随机种子
from stable_baselines3.common.vec_env import DummyVecEnv  # 导入DummyVecEnv用于向量化环境

matplotlib.use("Agg")  # 设置matplotlib后端为Agg（无GUI环境下绘图）


class StockTradingEnv(gym.Env):  # 定义股票交易环境，继承自gym.Env
    """
    A stock trading environment for OpenAI gym

    Parameters:
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
    """

    metadata = {"render.modes": ["human"]}  # 定义渲染模式

    def __init__(
        self,
        df: pd.DataFrame,  # 股票数据
        stock_dim: int,  # 股票数量
        hmax: int,  # 每次最大交易股数
        initial_amount: int,  # 初始资金
        num_stock_shares: list[int],  # 每只股票初始持仓
        buy_cost_pct: list[float],  # 买入手续费
        sell_cost_pct: list[float],  # 卖出手续费
        reward_scaling: float,  # 奖励缩放系数
        state_space: int,  # 状态空间维度
        action_space: int,  # 动作空间维度
        tech_indicator_list: list[str],  # 技术指标列表
        turbulence_threshold=None,  # 动荡阈值
        risk_indicator_col="turbulence",  # 风险指标列名
        make_plots: bool = False,  # 是否绘图
        print_verbosity=10,  # 打印频率
        day=0,  # 当前天数
        initial=True,  # 是否为初始状态
        previous_state=[],  # 上一状态
        model_name="",  # 模型名
        mode="",  # 模式
        iteration="",  # 迭代次数

        window_size=10, 
        feature_cols=None
    ):
        
        self.window_size = window_size
                
        # self.day = day  # 当前天数
        self.day = self.window_size  # 保证有足够历史

        self.df = df  # 股票数据
        self.stock_dim = stock_dim  # 股票数量
        self.hmax = hmax  # 每次最大交易股数
        self.num_stock_shares = num_stock_shares  # 每只股票初始持仓
        self.initial_amount = initial_amount  # 初始资金
        self.buy_cost_pct = buy_cost_pct  # 买入手续费
        self.sell_cost_pct = sell_cost_pct  # 卖出手续费
        self.reward_scaling = reward_scaling  # 奖励缩放系数
        self.state_space = state_space  # 状态空间维度
        self.action_space = action_space  # 动作空间维度
        self.tech_indicator_list = tech_indicator_list  # 技术指标列表
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))  # 连续动作空间，范围[-1,1]
        # self.observation_space = spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(self.state_space,)
        # )  # 观测空间，无穷范围

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window_size,self.state_space,)
        )  # 观测空间，无穷范围




        self.data = self.df.loc[self.day, :]  # 当前天的数据
        self.terminal = False  # 是否终止
        self.make_plots = make_plots  # 是否绘图
        self.print_verbosity = print_verbosity  # 打印频率
        self.turbulence_threshold = turbulence_threshold  # 动荡阈值
        self.risk_indicator_col = risk_indicator_col  # 风险指标列名
        self.initial = initial  # 是否为初始状态，（没用到）
        self.previous_state = previous_state  # 上一状态，（没用到）
        self.model_name = model_name  # 模型名
        self.mode = mode  # 模式
        self.iteration = iteration  # 迭代次数
        # 初始化状态
        self.state = self._initiate_state()
        

        # 初始化奖励、动荡、成本、交易次数
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0  # 回合数
        # 记录资产变化
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[-1][1 : 1 + self.stock_dim])
            )
        ]  # 初始总资产=现金+持仓市值
        self.rewards_memory = []  # 奖励历史
        self.actions_memory = []  # 动作历史
        self.state_memory = []  # 状态历史
        self.date_memory = [self._get_date()]  # 日期历史
        # self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()  # 初始化随机种子

        

          # episode ,记录
        self.history = None
        self.record_reward = None  # 记录奖励
        self.net_worth = None  # 记录净值
        # self.current_data = None  # 当前数据
        # self.obs_data = []  # 用于存储观测数据

    def _get_info(self):
        """获取当前的状态信息"""
    

        info = {            
          
            
            'total_net_worth': self.net_worth, # 当前净值        
            'total_profit': self.net_worth - self.initial_amount,  # 总利润
            # 'total_reward': self.net_worth - self.prev_net_worth ,  # 本步奖励   
            'total_reward': self.record_reward ,  # 本步奖励   
            # 'current_data': self.df.loc[self.day]['date'].strftime('%Y-%m-%d') , # 当前日期
            'current_data': self.df.loc[self.day].iloc[0]['date'],

            'total_shares': list(self.state[-1][(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) ,  # 持仓股数
            
            # 新加
            'total_trades': self.trades,  # 总交易次数

        }
        return info
    
    def _update_history(self, info):
        if not self.history:
            self.history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.history[key].append(value)


    def _sell_stock(self, index, action):
        # 卖出股票的内部方法
        def _do_sell_normal():
            if (
                self.state[-1][index + 2 * self.stock_dim + 1] != True
            ):  # 检查股票是否可卖
                if self.state[-1][index + self.stock_dim + 1] > 0:
                    # 当前持仓大于0才可卖
                    sell_num_shares = min(
                        abs(action), self.state[-1][index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[-1][index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # 更新现金
                    self.state[-1][0] += sell_amount

                    self.state[-1][index + self.stock_dim + 1] -= sell_num_shares
                    # 计算本次卖出交易的佣金成本：股票价格 × 卖出数量 × 卖出费率
                    self.cost += (
                        self.state[-1][index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # 动荡阈值判断
        if self.turbulence_threshold is not None:
            # 检查市场波动是否超过阈值
            if self.turbulence >= self.turbulence_threshold:
                # 检查股票价格是否大于0
                if self.state[-1][index + 1] > 0:
                    # 检查当前股票持仓是否大于0
                    if self.state[-1][index + self.stock_dim + 1] > 0:
                        # 卖出全部持仓
                        sell_num_shares = self.state[-1][index + self.stock_dim + 1]
                        # 计算卖出金额（考虑卖出手续费）
                        sell_amount = (
                            self.state[-1][index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # 更新现金余额（增加卖出金额）
                        self.state[-1][0] += sell_amount
                        # 将持仓数量清零
                        self.state[-1][index + self.stock_dim + 1] = 0
                        # 计算并累加交易成本
                        self.cost += (
                            self.state[-1][index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        # 交易次数加1
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        # 买入股票的内部方法
        def _do_buy():
            if (
                self.state[-1][index + 2 * self.stock_dim + 1] != True
            ):  # 检查股票是否可买
                available_amount = self.state[-1][0] // (
                    self.state[-1][index + 1] * (1 + self.buy_cost_pct[index])
                )  # 计算可买股数
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[-1][index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[-1][0] -= buy_amount
                self.state[-1][index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[-1][index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")  # 绘制资产曲线
        plt.savefig(f"results/account_value_trade_{self.episode}.png")  # 保存图片
        plt.close()  # 关闭绘图

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1  # 判断是否到最后一天
        if self.terminal:
            # 回合结束，输出统计信息
            if self.make_plots:
                self._make_plot()
            # 总资产 = 现金 + ∑(每只股票价格 × 持有数量)
            end_total_asset = self.state[-1][0] + sum(
                np.array(self.state[-1][1 : (self.stock_dim + 1)])
                * np.array(self.state[-1][(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.net_worth = end_total_asset  # 计算当前净值(加)

            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[-1][0]
                + sum(
                    np.array(self.state[-1][1 : (self.stock_dim + 1)])
                    * np.array(
                        self.state[-1][(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    )
                )
                - self.asset_memory[0]
            )  # 总奖励=最终总资产-初始资产
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    "results/actions_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                df_total_value.to_csv(
                    "results/account_value_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                df_rewards.to_csv(
                    "results/account_rewards_{}_{}_{}.csv".format(
                        self.mode, self.model_name, self.iteration
                    ),
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    "results/account_value_{}_{}_{}.png".format(
                        self.mode, self.model_name, self.iteration
                    )
                )
                plt.close()
            self.record_reward=self.reward  # 记录奖励
            self._update_history(self._get_info()) # 记录交易信息


             # reward = float(reward)  # 确保是 float
            if not np.isfinite(self.reward):
                raise ValueError(f"step有异常Reward has NaN/Inf")

            if not np.isfinite(self.state).all():
                raise ValueError(f"step有异常Next state has NaN/Inf")
            
            # 检查 action 是否正常
            action_arr = np.array(actions, dtype=np.float32)

            if not np.isfinite(action_arr).all():
                raise ValueError(f"step有异常Invalid action (NaN/Inf detected): {action_arr}")

            # return self.state, self.reward, self.terminal, False, {}
            return self.state, float(self.reward), self.terminal, False, {}

        else:
            actions = actions * self.hmax  # 动作缩放到最大交易量
            actions = actions.astype(
                int
            )  # 转为整数，不能买卖小数股
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)  # 动荡大时强制清仓
            begin_total_asset = self.state[-1][0] + sum(
                np.array(self.state[-1][1 : (self.stock_dim + 1)])
                * np.array(self.state[-1][(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # 对动作数组进行排序，返回排序后的索引数组（从小到大）
            argsort_actions = np.argsort(actions) 
            
            # 获取卖出动作的索引：取排序后数组的前N个（N=负值动作的数量）
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            
            # 获取买入动作的索引：将排序数组反转后取前M个（M=正值动作的数量）
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
               

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            self.actions_memory.append(actions)

            # 状态转移
            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[-1][0] + sum(
                np.array(self.state[-1][1 : (self.stock_dim + 1)])
                * np.array(self.state[-1][(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())

            # # ------------------- 奖励函数修改开始：改成收益率而非净值 -------------------
            # # 奖励为对数收益 (Log Return) 以防止奖励随净值增大而衰减
            # # self.reward = end_total_asset - begin_total_asset  # 原始奖励：资产增量
            # # 确保分母不为零，虽然在正常情况下 begin_total_asset 不应为零或负数
            # if begin_total_asset > 0:
            #     self.reward = np.log(end_total_asset / begin_total_asset)  # 奖励：log return
            # else:
            #     # 如果资产为零或负数，设置一个很大的负奖励，并终止回合
            #     self.reward = -100 # 大幅惩罚
            #     self.terminal = True
            # # ------------------- 奖励函数修改结束 -------------------

            self.reward = end_total_asset - begin_total_asset  # 奖励为资产增量
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling  # 奖励缩放
            self.state_memory.append(
                self.state
            )  # 保存当前状态

            self.net_worth = end_total_asset  # 计算当前净值(加)
            self.record_reward=self.reward  # 记录奖励
            self._update_history(self._get_info()) # 记录交易信息


        # reward = float(reward)  # 确保是 float
        if not np.isfinite(self.reward):
            raise ValueError(f"step有异常Reward has NaN/Inf")

        if not np.isfinite(self.state).all():
            raise ValueError(f"step有异常Next state has NaN/Inf")
        
        # 检查 action 是否正常
        action_arr = np.array(actions, dtype=np.float32)

        if not np.isfinite(action_arr).all():
            raise ValueError(f"step有异常Invalid action (NaN/Inf detected): {action_arr}")


        # return self.state, self.reward, self.terminal, False, {}
        return self.state, float(self.reward), self.terminal, False, {}

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # 初始化状态
        # self.day = 0
        self.day = self.window_size  # 初始从 window_size 开始（保证有足够历史）


        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[-1][1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[-1][0] + sum(
                np.array(self.state[-1][1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[-1][(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        self.history = None
        self.record_reward=0  # 记录奖励
        

        

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state  # 返回当前状态

 

    def _initiate_state(self):
        obs_window = []

        

        # 包含当前 step 的 window
        start = self.day - self.window_size + 1
        end = self.day
        window_data = self.df.loc[start:end]  # 这样包含 self.day 这一行

        for i in range(len(window_data.index.unique())):
            data_i = window_data.loc[window_data.index.unique()[i]]

            if len(self.df.tic.unique()) > 1:
                # 多股票
                state_i = (
                    [self.initial_amount] +
                    data_i.close.values.tolist() +
                    self.num_stock_shares +
                    sum(
                        (data_i[tech].values.tolist() for tech in self.tech_indicator_list),
                        []
                    )
                )
            else:
                # 单股票
                state_i = (
                    [self.initial_amount] +
                    [data_i.close] +
                    [0] * self.stock_dim +
                    sum(([data_i[tech]] for tech in self.tech_indicator_list), [])
                )

            obs_window.append(state_i)

        state_arr = np.array(obs_window, dtype=np.float32)
        if not np.isfinite(state_arr).all():
            raise ValueError(f"初始化状态中有空值State has NaN/Inf:\n{state_arr}")
        
        # raise ValueError(f"测试异常")


        return np.array(obs_window, dtype=np.float32)


    

    def _update_state(self):
        obs_window = []

        # start = self.day - self.window_size
        # end = self.day

        # window_data = self.df.loc[start:end - 1]

        # 包含当前 step 的 window
        start = self.day - self.window_size + 1
        end = self.day
        window_data = self.df.loc[start:end]  # 这样包含 self.day 这一行

        for i in range(len(window_data.index.unique())):
            data_i = window_data.loc[window_data.index.unique()[i]]

            if len(self.df.tic.unique()) > 1:
                state_i = (
                    [self.state[-1][0]] +
                    data_i.close.values.tolist() +
                    list(self.state[-1][(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                    sum(
                        (data_i[tech].values.tolist() for tech in self.tech_indicator_list),
                        []
                    )
                )
            else:
                state_i = (
                    [self.state[-1][0]] +
                    [data_i.close] +
                    list(self.state[-1][(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
                    sum(([data_i[tech]] for tech in self.tech_indicator_list), [])
                )

            obs_window.append(state_i)

        state_arr = np.array(obs_window, dtype=np.float32)
        if not np.isfinite(state_arr).all():
            raise ValueError(f"更新状态中有空值State has NaN/Inf:\n{state_arr}")

        # raise ValueError(f"测试异常")
        

        return np.array(obs_window, dtype=np.float32)


    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # 保存状态历史
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    # 保存资产历史
    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    # 保存动作历史
    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])  # 包装为DummyVecEnv
        obs = e.reset()
        return e, obs
    
    
    



if __name__ == "__main__":

# 1、数据处理

    import pandas as pd

    # 读取训练集数据
    # train = pd.read_csv('data_FinRL/train_data.csv', parse_dates=True) # 注意这里的parse_dates=True可以自动解析日期列

    train = pd.read_csv('data_FinRL/train_data.csv')
    trade = pd.read_csv('data_FinRL/trade_data.csv')

    # If you are not using the data generated from part 1 of this tutorial, make sure 
    # it has the columns and index in the form that could be make into the environment. 
    # Then you can comment and skip the following lines.
    train = train.set_index(train.columns[0])
    train.index.names = ['']
    trade = trade.set_index(trade.columns[0])
    trade.index.names = ['']

    # 查看前几行
    print(train.head())
    print(trade.head())

    # stockstats technical indicator column names
    # check https://pypi.org/project/stockstats/ for different names
    INDICATORS = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    stock_dimension = len(train.tic.unique())
    # 状态空间的维度 = 账户信息（现金余额或总资产，是全局特征） + 每只股票的基本特征（价格、持仓） + 每只股票的所有技术指标特征。
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension


    env_kwargs = {
        "hmax": 100,
        "initial_amount": 10000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }

    # import env_stocktrading_FinRL as StockTradingEnv
    # from env_stocktrading_FinRL import StockTradingEnv
    # from env_dou.Double2_finrl import StockTradingEnv


    e_train_gym = StockTradingEnv(df = train, **env_kwargs)

    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train)) #<class 'stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv'>

# 2、训练测试

    obs = env_train.reset()
    print('状态值：',obs.shape, obs)  # 打印观测的形状和内容 状态值： (1, 4) [[ 0.0478776  -0.03677095 -0.00638026  0.04655835]]
    # quit()
    from stable_baselines3 import PPO, A2C, DDPG

    model = PPO("MlpPolicy", env_train, verbose=1)
    model.learn(total_timesteps=5000)

    # *******************2、测试********************

    test_env = StockTradingEnv(df = trade, **env_kwargs)
    test_env, _ = test_env.get_sb_env()
    obs = test_env.reset()


    total_reward = 0
    all_actions = {"0":0,"1":0,"2":0,"total_reward":[]}  # 用于存储所有动作
    actions_list = []
    for i in range(5000):
    # for i in range(2):
        action, _state = model.predict(obs, deterministic=True) # 当 deterministic=True 时，模型每次在相同状态下都会输出同一个动作（即选择概率最大的动作），结果是可复现的，常用于测试和评估。
        # action, _state = model.predict(obs, deterministic=False) # 
        # print(f"Action space: {env_test.action_space}")  # Discrete(2) - left or right
        # print(f"Action space shape: {env_test.action_space.shape}")  # Discrete(2) - left or right
        print("Action shape:", type(action),action.shape) #Action shape: (1, 2)
        print("Action value:", action)     # Action value: [[0.06333294 0.        ]]
        print(f"Action: {action[0][0], action[0][1]}")

        historypre=test_env.envs[0].history

        obs, reward, done, info = test_env.step(action)
        # print(f"Step {i+1}: Action taken: {action}, obs:{obs.shape},Reward: {reward}, Done: {done}, Info: {info}")
        # # print(f"Step {i}, Action: {action[0]}, Reward: {reward},")
        # print('打印类型信息',type(action), type(obs), type(reward), type(done), type(info))  # 打印类型信息

        # env_test.render("human")
        test_env.envs[0].render("human")
        # VecEnv resets automatically
        total_reward += reward[0]
        if done[0]:
            obs = test_env.reset()
            print(f"Episode finished after {i+1} timesteps")
            print(f"Test Episode {i+1}, Total Reward: {total_reward}")
            all_actions['total_reward'].append(total_reward)
            total_reward = 0

        actions_list.append(action)
        # print(f"Step {i+1}, Action: {action[0]}, Reward: {reward[0]}, Total Reward: {total_reward}, Done: {done[0]}, Info: {info}")

        # if action[0] == 1:
        #     all_actions['1'] +=1
        # elif action[0] == 2:
        #     all_actions['2'] +=1
        # else:       
        #     all_actions['0'] +=1
    print("所有动作统计：", all_actions)  # 打印所有动作的统计信息

    


    # 1、多股小数
    import numpy as np
    from collections import Counter
    # 假设最大交易量
    hmax = 100  # e.g., 100股最大交易

    # 👉 步骤1：放大动作
    actions_scaled = [a * hmax for a in actions_list]

    # 👉 步骤2：四舍五入后转为 tuple（便于比较、计数）
    # action_tuples = [tuple(np.round(a.flatten(), 2)) for a in actions_scaled] #保留2位

    action_tuples = [tuple(np.round(a.flatten())) for a in actions_scaled]

    # 👉 步骤3：计数重复
    action_counts = Counter(action_tuples)

    # 👉 输出统计结果
    # for action, count in action_counts.items():
    #     print(f"Action: {np.array(action)} -> Count: {count}")
    
    print("动作次数最多的统计：")
    top_actions = action_counts.most_common(6)
    # 打印统计结果
    for action, count in top_actions:
        print(f"Action: {action} -> Count: {count}")



    print("total_reward 统计：", len(historypre['total_reward']), historypre['total_reward'][-5:])  # 打印前5个总奖励
    print("total_shares 统计：", len(historypre['total_shares']), historypre['total_shares'][-5:])  # 打印前5个总奖励



