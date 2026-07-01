# Import libraries
import pandas as pd  # Import pandas for data processing
import numpy as np  # Import numpy for numerical calculations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import yfinance as yf  # Import yfinance to fetch financial data
from stable_baselines3 import PPO  # Import PPO algorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize  # Import environment wrappers
from stable_baselines3.common import env_checker  # Import environment checker tool
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy  # Import policy base classes
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # Import feature extractor base class
import gymnasium as gym  # Import gymnasium for custom environments
from gymnasium import spaces  # Import spaces for defining observation/action spaces
import torch  # Import PyTorch
import torch.nn as nn  # Import PyTorch neural network module
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
import gc

# from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
# from stable_baselines3.common.logger import HParam

# -------------------------
# 1. PPO Hyperparameters
# -------------------------
Features_Dim = 512  # Dimension of the feature extractor
BATCH_SIZE = 64  # Batch size
# Total_Timesteps = 100  # Total training steps / total sampling iterations
Total_Timesteps = 10_0000  # Total training steps / total sampling iterations
LR = 3e-4   # Learning rate
Ent_coef = 0.01   # Entropy coefficient
N_steps = 64   # Number of steps per training iteration (sampling steps)
N_epochs = 10   # Number of epochs for updates per sampling phase


# -------------------------
# 2. Evaluation Hyperparameters
# -------------------------
WINDOW_SIZE = 64  # Time window size
Threshold = 0.8  # Reward threshold for saving the model
N_tests = 300

Recurren_ppo = None
Reppo_lstm_hidden_size = 256
Use_PVcache = False # Whether to use PV cache (Apply to both model and logic)


# 1. ====== Custom Packages ======

# Custom utilities
from utilss import metrics_plot
from utilss.log_util import EvalLogger  # Import logger
import utilss.date_util as date_util
from utilss.tb_csv_util import export_to_csv_unique_top

# Simulated stock market environment
from stock_env.Stock_env import StockTradingEnv as TradingEnv

# PyramidMambaMoE architecture related packages
from moduls.CustomPPOmodel import PPO_FeatureExtractor as PPO_FeatureExtractor
from moduls.CustomPPOmodel import PPO_Policy as PPO_CustomNetwork
torch.autograd.set_detect_anomaly(True)

# Function to read data and set parameters
def get_data_params(data_path):
    # 1. Data Processing
    import pandas as pd
    
    # get_data = pd.read_csv(f'./data_test/processed_full.csv')
    get_data = pd.read_csv(data_path)
    get_data = get_data.set_index(get_data.columns[0])
    get_data.index.names = ['']
    # print(get_data.head())

    # Check for NaN or Inf
    numeric_df = get_data.select_dtypes(include=[np.number])
    has_nan = get_data.isna().any().any()
    has_inf = np.isinf(numeric_df.to_numpy()).any()

    print("Has NaN:", has_nan)
    print("Has Inf:", has_inf)

    # Data splitting (with window)
    TRAIN_START_DATE = '2020-01-01'
    TRAIN_END_DATE = '2023-01-01'
    TRADE_START_DATE = '2023-01-01'
    TRADE_END_DATE = '2025-01-01'

    training_data = date_util.data_split_with_window(get_data, TRAIN_START_DATE, TRAIN_END_DATE, window_size=0)
    test_data = date_util.data_split_with_window(get_data, TRADE_START_DATE, TRADE_END_DATE, window_size=WINDOW_SIZE)

    print("\nTrain length:", len(training_data))
    print(training_data.tail(20))

    print("\nTrade length (with window):", len(test_data))
    print(test_data.head(20))

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

    stock_dimension = len(training_data.tic.unique())
    # State space dimension = Account Info (Balance/Assets) + Basic features per stock (Price, Shares) + Indicators per stock
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension


    # Set environment parameters
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        'window_size': WINDOW_SIZE, 
    }
    return training_data, test_data, env_kwargs


# 2. Evaluation function, evaluating model on test set
def test_Agent(vec_env, model, n_tests=3000, n_top=10, save_prefix="./logs/test_results"):
    n_tests = N_tests   # ✅ Re-assignment
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)  # Create multi-level directory
    
    # Reset environment before testing to get initial observation
    obs = vec_env.reset()  # Environment type: DummyVecEnv

    if Use_PVcache:
        # Call reset_cache method
        model.policy.mlp_extractor.reset_Icache()
        print("Inference done, testing started, feature cache reset.")

    total_reward = 0
    all_actions = {"total_reward": []}  # Used to store all actions
    actions_list = []
    for i in range(n_tests):
        action, _state = model.predict(obs, deterministic=True)
        historypre = vec_env.envs[0].history        
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]

        if done[0] or i == n_tests - 1:
            all_actions['total_reward'].append(float(total_reward))
            total_reward = 0
            break  # End loop
        actions_list.append(action)

    all_actions['actions_list'] = actions_list

    # 1. Multi-stock decimal handling
    import numpy as np
    from collections import Counter
    import json
    
    # Assume max trading volume
    hmax = 100  # e.g., 100 shares max trade

    # 👉 Step 1: Scale actions
    actions_scaled = [a * hmax for a in actions_list]

    # 👉 Step 2: Round and convert to tuple (for comparison and counting)
    action_tuples = [tuple(np.round(a.flatten())) for a in actions_scaled]

    # 👉 Step 3: Count repetitions
    action_counts = Counter(action_tuples)
    print("Action frequency statistics:", action_counts.total())
    top_actions = action_counts.most_common(n_top)
    
    # Print statistics
    for action, count in top_actions:
        print(f"Action: {action} -> Count: {count}")

    # 2. Position/Holding shares
    # Convert to tuple then count
    total_shares = Counter(tuple(x) for x in historypre['total_shares'])
    shares_counts = Counter(total_shares)
    print("Holding frequency statistics:", shares_counts.total())
    top_shares = shares_counts.most_common(n_top)
    
    # Print statistics
    for action, count in top_shares:
        print(f"Action: {action} -> Count: {count}")

    all_actions['top_actions'] = top_actions
    all_actions['top_shares'] = top_shares
    summary_values = metrics_plot.calculate_metrics_values({'PPO_Lstm': historypre})  # Generate overall metrics summary
    all_actions['summary_values'] = summary_values
    
    # === Merge into a single dict ===
    combined_result = {
        "history": historypre,
        "actions_summary": all_actions
    }

    # === JSON Serialization Fix ===
    def default_serializer(o):
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        elif isinstance(o, (np.floating,)):
            return float(o)
        elif isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)

    # === Save to jsonl ===
    with open(f"{save_prefix}.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps(combined_result, ensure_ascii=False, default=default_serializer) + "\n")

    # === Save to csv ===
    df = pd.json_normalize(combined_result)
    df.to_csv(f"{save_prefix}.csv", index=False)
    return historypre, all_actions


# 3. Custom callback function, recording training data and evaluating model after each batch
class SimpleStockEvalCallback(BaseCallback):
    """
    - Evaluates once after each batch update
    - Gets net worth list and calculates return rate
    - Saves the best model and models exceeding threshold
    - Logs return rate to the logger
    """
    def __init__(self, eval_env, save_path="./logs/eval_model", metric_func=None, reward_threshold=None, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.save_path = save_path
        self.reward_threshold = reward_threshold
        self.best_return = -np.inf
        self.metric_func = metric_func
        os.makedirs(f'{self.save_path}/eval_model', exist_ok=True)

    def _on_step(self) -> bool:
        # 1. PV cache reset
        if Use_PVcache:
            done = self.locals["dones"][0]
            print(f"Done Status: {type(done)}, {done}")
            
            # If the environment has ended
            if done:
                self.model.policy.mlp_extractor.reset_Tcache()
                print("Training done, training feature cache reset.")
                V_T, P_T, P_I = self.model.policy.mlp_extractor.get_cache_shape()
                print(f"Reset cache shape, V_T: {V_T}, P_T: {P_T}, P_I: {P_I}")
                
        # 2. Evaluate model every n_steps (after training a batch of samples)
        if self.num_timesteps > 0 and self.num_timesteps % self.model.n_steps == 0:
            historypre, all_actions = self.metric_func(vec_env=self.eval_env, model=self.model, 
                                                       save_prefix=f'{self.save_path}/eval_test/eval_{self.num_timesteps}',
                                                       n_tests=N_tests, n_top=10)
          
            ret = all_actions['summary_values'][0]['收益率'] 
            max_ret = all_actions['summary_values'][0]['最大收益率'] # Maximum return rate
           
            # Log to logger
            self.logger.record("eval/return_rate", ret)
            self.logger.record("eval/return_maxrate", max_ret)
            self.logger.record("eval/step", self.num_timesteps)

            if self.verbose > 0:
                print(f"[Step {self.num_timesteps}] Return Rate: {ret:.2%}, Max Return: {max_ret:.2%}")

            # Save the best model
            if ret > self.best_return:
                self.best_return = ret
                best_path = os.path.join(f'{self.save_path}/eval_model', "best_model")                
                self.logger.record("eval/custom_best", ret)
                self.logger.record("eval/custom_best_step", self.num_timesteps)
                if self.verbose > 0:
                    print(f"  >> Saving new best model (Return={ret:.2%}) → {best_path}")

            # Save when threshold is reached
            if self.reward_threshold is not None and ret >= self.reward_threshold:
                thres_path = os.path.join(f'{self.save_path}/eval_model', f"model_step_{ret:.0%}_{self.num_timesteps}")
                self.model.save(thres_path)                
                self.logger.record("eval/custom_threshold", ret)
                self.logger.record("eval/custom_thre_step", self.num_timesteps)
                if self.verbose > 0:
                    print(f"  >> Reward threshold {self.reward_threshold:.2%} reached, saving model → {thres_path}")

        return True

from datetime import datetime
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# base_dir = f'./logs_ppo_gate/{timestamp}/'
base_dir = f'./logs_ppo_test_gate/{timestamp}/'


# Model Training Function
def train(data_path):
    global Recurren_ppo, TradingEnv, PPO_FeatureExtractor, PPO_CustomNetwork, PPO
    # Fetch data
    training_data, test_data, env_kwargs = get_data_params(data_path)

    # Configure feature extractor
    policy_kwargs = dict(
        features_extractor_class=PPO_FeatureExtractor,
        features_extractor_kwargs=dict(features_dim=Features_Dim),    
    )

    if Recurren_ppo: # If Recurren_ppo, add lstm_hidden_size
        policy_kwargs = dict(
            features_extractor_class=PPO_FeatureExtractor,
            features_extractor_kwargs=dict(features_dim=Features_Dim),
            lstm_hidden_size = Reppo_lstm_hidden_size,    
        )

    if PPO_FeatureExtractor is None: # Use standard SB3 model
        policy_kwargs = None

    # 0. Logger setup
    os.makedirs(f'{base_dir}/charts', exist_ok=True) 
    log_path = os.path.join(base_dir, f"ppo_log_{timestamp}.txt")
    logger = EvalLogger(log_path=log_path)  # Initialize logger
    logger.info(f"Eval logger initialized ✅, Time: {timestamp}")

    # 1. Environment creation
    # Create environment instance
    train_env = TradingEnv(df=training_data, **env_kwargs)

    # Check environment
    env_checker.check_env(train_env)  # Ensure environment follows SB3 specs

    print(train_env.action_space)  # Print action space Discrete(3) - hold, buy, sell
    print(train_env.observation_space)  # Print observation space
    
    # Check observation dimensions
    obs, info = train_env.reset()
    print('Obs shape:', obs.shape, obs)  # Print observation shape and content

    # Wrap environment
    train_env = DummyVecEnv([lambda: train_env])  # Wrap as vectorized environment
    train_env.reset()

    # Check environment for NaNs
    train_env = VecCheckNan(train_env, raise_exception=True)

    # Test environment setup
    test_env_vec = DummyVecEnv([lambda: TradingEnv(test_data, **env_kwargs)])  # Vectorized test environment
    obs = test_env_vec.reset()


    # 2. Model Training
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using: {device} device")

    PPO_model = PPO(PPO_CustomNetwork, train_env, policy_kwargs=policy_kwargs, 
                    verbose=1, batch_size=BATCH_SIZE, tensorboard_log=f'{base_dir}/ppo/',
                    ent_coef=Ent_coef, learning_rate=LR, device=device, n_steps=N_steps, n_epochs=N_epochs)
    
    # Custom logger configuration
    new_logger = configure(f'{base_dir}/ppo/', ["stdout", "csv", "log", "tensorboard", "json"])
    PPO_model.set_logger(new_logger)

    # Callback
    callback = SimpleStockEvalCallback(eval_env=test_env_vec, reward_threshold=Threshold, verbose=1,
                                       save_path=f'{base_dir}', metric_func=test_Agent)

    PPO_model.learn(total_timesteps=Total_Timesteps, callback=callback, progress_bar=True)  # Train model
    train_env.close()

    # 3. Model Testing
    PPO_history, PPO_all_actions = test_Agent(test_env_vec, PPO_model, n_tests=N_tests, save_prefix=f'{base_dir}/test_results/test')
    test_env_vec.close()

    # 4. Print information
    print('*'*50, 'PPO_Lstm', '*'*50)
    print("Action statistics:", PPO_all_actions)  # Action statistics
    print("History keys:", PPO_history.keys())  
    print("Net worth stats:", len(PPO_history['total_net_worth']), PPO_history['total_net_worth'][-5:])  
    print("Profit stats:", PPO_history['total_profit'][-5:])  
    print("Reward stats:", PPO_history['total_reward'][-5:])  
    print("Current data stats:", PPO_history['current_data'][-5:])  
    print('Total trades stats:', PPO_history['total_trades'])  
    logger.log_model_section(model_name='PPO2_Lstm', all_actions=PPO_all_actions, history_pre=PPO_history)

    # 5. Plotting
    history_dict = {'PPO_Lstm': PPO_history}
    data_dict = metrics_plot.calculate_metrics_series(history_dict)  # Generate time series and metrics table
    summary_values = metrics_plot.calculate_metrics_values(history_dict)  # Generate overall metrics summary
    print('*'*50, 'Evaluation Metrics', '*'*50)
    print(summary_values)
    logger.log_metrics_section(metrics=summary_values)  # Log metrics summary
    metrics_plot.plot_metrics_and_summary(data_dict, summary_values, base_path=f"{base_dir}/charts", show=False)
    metrics_plot.matplot_show(PPO_history, base_path=f"{base_dir}/charts", show=False, data=False)
    print(PPO_model.policy)  # Print policy network structure

    # 6. Print model structure and parameter info
    print("="* 50)
    print("Model Summary:")
    from torchinfo import summary

    vec_env = PPO_model.get_env()
    obs = vec_env.reset()
    print('Class name:', PPO_model.policy.__class__.__name__)
    PPO_name = PPO_model.policy.__class__.__name__
    if PPO_name != 'RecurrentActorCriticPolicy':
        summary(PPO_model.policy, input_size=obs.shape)

    print("="* 50)
    resul_best = export_to_csv_unique_top(f'{base_dir}/ppo', f'{base_dir}/ppo',
                                         metrics_col="eval/custom_threshold", best_metrics_col="eval/custom_best", max_return_col='eval/return_maxrate',
                                         top_k=5, mode="max")
    print(resul_best)
    print("="* 50)
    resul_best['threshold'] = Threshold    
    
    # =========================================================
    # 🌟 Parameter Statistics 🌟
    # =========================================================
    # 1. Count total parameters
    total_params = sum(p.numel() for p in PPO_model.policy.parameters())

    # 2. Convert to Million (M) and Billion (B) units
    params_M = total_params / 1_000_000
    params_B = total_params / 1_000_000_000

    # 3. Print results
    print("Model parameter count:")
    print(f"Total parameters (Total): {total_params:,}")
    print(f"Parameters (Million - M): {params_M:.2f} M")
    print(f"Parameters (Billion - B): {params_B:.5f} B")
    resul_best['params_M'] = params_M
    resul_best['params_B'] = params_B
    logger.info(f"Parameters (Million - M): {params_M:.2f} M; Parameters (Billion - B): {params_B:.5f} B")


    # 7. Release VRAM and memory

    # =========================================================
    # 🌟 Memory/VRAM Release Area (Clearing current process only) 🌟
    # =========================================================
    # 1. Delete model and environment instances to clear references
    if 'PPO_model' in locals():
        del PPO_model
        print("Deleted PPO_model")
    # env.close has been executed; just delete reference here
    if 'train_env' in locals():
        train_env.close()
        del train_env
        print("Deleted train_env")
    if 'test_env_vec' in locals():
        test_env_vec.close()
        del test_env_vec
        print("Deleted test_env_vec")
        
    # 2. Clear global references
    Recurren_ppo = TradingEnv = PPO_FeatureExtractor = PPO_CustomNetwork = PPO = None 

    # 3. Force Python garbage collection
    gc.collect()

    # 4. **Clear CUDA cache for current process only**
    # This won't affect GPU memory allocated by other processes
    if torch.cuda.is_available():
        print("Clearing CUDA cache...")
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")
    
    # Optional: Print memory usage (for debugging)
    if torch.cuda.is_available():
        print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated, "
            f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB reserved")

    return resul_best


if __name__ == "__main__":
    dataset_path = './dataset/DJIA_2025.csv' # data 2016.01~2025.10
    train(dataset_path)



    