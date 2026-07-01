import logging
import os
from datetime import datetime
from typing import Dict, List, Any


class EvalLogger:
    def __init__(self, log_path: str = None):
        # 默认日志路径为 ./logs/log_当前时间.txt
        if log_path is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            log_dir = './logs'
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"log_{timestamp}.txt")

        self.log_path = log_path
        self.logger = logging.getLogger(f"EvalLogger_{id(self)}")
        self.logger.setLevel(logging.INFO)

        # 防止多次初始化 Handler
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_path, encoding='utf-8')
            ch = logging.StreamHandler() # 同时也在控制台输出
            formatter = logging.Formatter('%(message)s')
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

        # ✅ 只添加文件输出 handler，不输出到控制台
        # if not self.logger.handlers:
        #     # ✅ 只添加文件输出 handler，不输出到控制台
        #     fh = logging.FileHandler(self.log_path, encoding='utf-8')
        #     formatter = logging.Formatter('%(message)s')
        #     fh.setFormatter(formatter)
        #     self.logger.addHandler(fh)

    def info(self, msg: str):
        self.logger.info(msg)
        
    def error(self, msg: str):
        self.logger.error(msg)
    def log_model_section(self, model_name: str, all_actions: Dict, history_pre: Dict[str, Any]):
        self.logger.info(f"{'*' * 50} {model_name} {'*' * 50}")
        self.logger.info(f"all_actions 统计： {all_actions}")
        self.logger.info(f"history pre统计： {history_pre.keys()}")
        self.logger.info(f"total_net_worth 统计： {len(history_pre['total_net_worth'])} {history_pre['total_net_worth'][-5:]}")
        self.logger.info(f"total_profit 统计： {history_pre['total_profit'][-5:]}")
        self.logger.info(f"total_reward 统计： {history_pre['total_reward'][-5:]}")
        self.logger.info(f"current_data 统计： {history_pre['current_data'][-5:]}")

        # print('total_trades 统计：', PPO_history['total_trades'])  # 打印前5个总交易次数
        self.logger.info(f"total_trades 统计： {history_pre['total_trades'][-5:]}")


    def log_metrics_section(self, metrics: List[Dict[str, Any]]):
        self.logger.info(f"{'*' * 50} 评估指标 {'*' * 50}")
        for metric in metrics:
            for k, v in metric.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info("-" * 80)

    def log_chart_paths(self, chart_html: str, mat_fig: str):
        self.logger.info(f"✅ 图表保存：{chart_html}")
        self.logger.info(f"✅ Mat图片保存：{mat_fig}")

    def log_line(self, msg: str):
        self.logger.info(msg)



if __name__ == "__main__":
    # === 设置日志输出路径 ===
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"my_log_{timestamp}.txt")

    # === 初始化 ===
    logger = EvalLogger()  # 或 EvalLogger(log_path="./mylogs/eval_ppo.txt")

    logger = EvalLogger(log_path=log_path)
    logger.info(f"Eval logger 初始化完成 ✅，时间：{timestamp}")

    
