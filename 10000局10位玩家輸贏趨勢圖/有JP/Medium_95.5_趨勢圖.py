import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 設定 matplotlib 字體以支援中文
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# ------------------------------
# 遊戲設定參數
# ------------------------------
positions = list(range(17))
probs = torch.tensor([
    0.0000152587890625, 0.000244140625, 0.0018310546875, 0.008544921875,
    0.02777099609375, 0.066650390625, 0.1221923828125, 0.174560546875,
    0.196380615234375, 0.174560546875, 0.1221923828125, 0.066650390625,
    0.02777099609375, 0.008544921875, 0.0018310546875, 0.000244140625,
    0.0000152587890625
], dtype=torch.float32)

payout_table = torch.tensor([
    200.0, 36.0, 0, 0, 2.0,
    1.3, 0.6, 0.4, 0.1, 0.4,
    0.6, 1.3, 2.0, 0, 0,
    36.0, 200.0
], dtype=torch.float32)

ball_multipliers = torch.tensor([1, 2, 5], dtype=torch.float32)
ball_probs = torch.tensor([0.93, 0.05, 0.02], dtype=torch.float32)

free_game_positions = {2, 3, 13, 14}
rounds_per_player = 10_000
num_players = 10
bet_amount = 1.0
jackpot_contribution_rate = 0.01
max_payout = 1000
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------
# 免費遊戲 mini slot
# ------------------------------
def draw_free_game_mini_slot(rng, jackpot_pool, bet_amount, multiplier):
    roll = torch.rand(1, device=device, generator=rng).cpu().item()
    if roll < 0.00221:
        jp_share = 0.25 * jackpot_pool * (bet_amount / max_payout)
        reward = (jp_share + bet_amount * 200) * multiplier
        return reward, "Super JP / X 200", jp_share
    elif roll < 0.01135:
        jp_share = 0.10 * jackpot_pool * (bet_amount / max_payout)
        reward = (jp_share + bet_amount * 100) * multiplier
        return reward, "Lucky JP / X 100", jp_share
    elif roll < 0.07568:
        return bet_amount * 50 * multiplier, "X 50", 0.0
    elif roll < 0.19027:
        return bet_amount * 25 * multiplier, "X 25", 0.0
    elif roll < 0.48916:
        return bet_amount * 10 * multiplier, "X 10", 0.0
    else:
        return bet_amount * 3 * multiplier, "X 3", 0.0

# ------------------------------
# 模擬單一玩家
# ------------------------------
def simulate_player(rng, rounds):
    probs_gpu = probs.to(device)
    payout_table_gpu = payout_table.to(device)
    ball_probs_gpu = ball_probs.to(device)
    multipliers_gpu = ball_multipliers.to(device)

    positions_drawn = torch.multinomial(probs_gpu, num_samples=rounds, replacement=True, generator=rng)
    multiplier_drawn = torch.multinomial(ball_probs_gpu, num_samples=rounds, replacement=True, generator=rng)
    multipliers = multipliers_gpu[multiplier_drawn]
    base_prizes = payout_table_gpu[positions_drawn]

    jackpot_pool = 0.0
    profit = 0.0
    cumulative_profit = []

    for i in range(rounds):
        jackpot_pool += bet_amount * jackpot_contribution_rate
        pos = positions_drawn[i].item()
        mult = multipliers[i].item()
        base_win = base_prizes[i].item() * mult

        if pos in free_game_positions:
            reward, _, jp_share = draw_free_game_mini_slot(rng, jackpot_pool, bet_amount, mult)
            jackpot_pool -= jp_share
        else:
            reward = 0.0

        total = base_win + reward
        profit += total - bet_amount
        cumulative_profit.append(profit)

    return cumulative_profit

# ------------------------------
# 主模擬與繪圖
# ------------------------------
torch.manual_seed(42)
rng = torch.Generator(device=device).manual_seed(42)

player_trends = []
for i in range(num_players):
    print(f"模擬第 {i + 1} 位玩家...")
    trend = simulate_player(rng, rounds_per_player)
    player_trends.append(trend)

# 繪製趨勢圖
plt.figure(figsize=(12, 6))
for i, trend in enumerate(player_trends):
    plt.plot(trend, label=f"玩家 {i + 1}")
plt.title("Medium_95.5_趨勢圖")
plt.xlabel("遊戲局數")
plt.ylabel("累積盈虧")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
