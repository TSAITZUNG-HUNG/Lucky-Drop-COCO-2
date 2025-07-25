import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# 遊戲設定參數
# ------------------------------
positions = list(range(17))
probs = [0.0000152587890625, 0.000244140625, 0.0018310546875, 0.008544921875,
         0.02777099609375, 0.066650390625, 0.1221923828125, 0.174560546875,
         0.196380615234375, 0.174560546875, 0.1221923828125, 0.066650390625,
         0.02777099609375, 0.008544921875, 0.0018310546875, 0.000244140625, 0.0000152587890625]

payout_table = {
    0: 60.0, 1: 30.0, 2: 8.0, 3: 2.6, 4: 1.6,
    5: 1.3, 6: 1.0, 7: 0.6, 8: 0.2, 9: 0.6,
    10: 1.0, 11: 1.3, 12: 1.6, 13: 2.6, 14: 8.0,
    15: 30.0, 16: 60.0
}

ball_multipliers = [1, 2, 5]
ball_probs = [0.93, 0.05, 0.02]
rounds_per_player = 10000
num_players = 10
bet_amount = 1

# ------------------------------
# 模擬單一玩家
# ------------------------------
def simulate_player(rng, rounds):
    profit = 0.0
    cumulative_profit = []

    for _ in range(rounds):
        pos = rng.choice(positions, p=probs)
        base_prize = payout_table[pos]
        multiplier = rng.choice(ball_multipliers, p=ball_probs)
        payout = base_prize * multiplier
        profit += payout - bet_amount
        cumulative_profit.append(profit)

    return cumulative_profit

# ------------------------------
# 模擬全部玩家並收集結果
# ------------------------------
rng = np.random.default_rng(42)
all_players_profits = []

for i in range(num_players):
    profits = simulate_player(rng, rounds_per_player)
    all_players_profits.append(profits)

# ------------------------------
# 畫出趨勢圖
# ------------------------------
plt.figure(figsize=(12, 6))
for i, profits in enumerate(all_players_profits):
    plt.plot(profits, label=f"Player {i+1}")
plt.title("Low_95.5_趨勢圖")
plt.xlabel("Rounds Played")
plt.ylabel("Cumulative Profit")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
