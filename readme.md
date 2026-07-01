# 1. Introduction

MSF-Trader is a reinforcement learning-based stock trading framework designed for complex financial markets with high noise, strong non-stationarity, and multi-scale temporal dynamics.

The framework combines:

* SwiGLU position-aware market representation learning
* Pyramid downsampling for multi-scale temporal feature extraction
* Mamba-MoE-based long-sequence state modeling
* Cross-attention-based multi-scale feature fusion
* DPA-SwiGLU-driven PPO decision optimization

to improve market state representation, long-sequence feature extraction efficiency, multi-scale information interaction, trading profitability, and decision stability.

The main highlights are:

* A SwiGLU position-aware embedding module is designed for market representation learning.
* A pyramid Mamba-MoE module is proposed to extract and fuse multi-scale temporal features.
* A DPA-SwiGLU policy network is developed to enhance PPO-based trading decisions.

# 2. How to Run

Step 1: Install the Environment

```bash
pip install -r requirements.txt
```

Step 2: Start Training

```bash
python main.py
```

# 3. Code Directory Structure

The modules corresponding to the code and paper are as follows:

```text
.
├── dataset                     # Datasets
├── main.py                     # Main program for training and testing; PPO training parameter settings
├── moduls
│   ├── CustomPPOmodel.py       # Custom PPO model; DPA-SwiGLU policy and value network settings
│   └── moduls                  # Framework modules
│       ├── Cross_atension.py   # Cross-attention module for multi-scale feature fusion
│       ├── Pyramid_MambaMoe.py # Pyramid Mamba-MoE module for multi-scale temporal feature extraction
│       ├── Share_Moe.py        # Shared MoE module
│       └── Swiglu_position.py  # SwiGLU position-aware embedding module
├── readme.md                   # Project documentation/README
├── stock_env
│   └── Stock_env.py            # Stock trading environment
└── utilss                      # Utilities: evaluation, testing, logging, and charting tools
    ├── date_util.py
    ├── log_util.py
    ├── metrics_plot.py
    └── tb_csv_util.py
```
