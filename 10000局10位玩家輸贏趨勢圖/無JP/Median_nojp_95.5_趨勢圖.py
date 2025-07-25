import torch
import pandas as pd
import matplotlib.pyplot as plt

# 設定中文字型（macOS 預設有的字型）
plt.rcParams['font.family'] = 'Heiti TC'  # 或 'PingFang TC'、'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號


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
bet_amount = 1.0
rounds_per_player = 10_000
num_players = 10
device = torch.device("cpu")  # 避免 GPU/MPS 爆記憶體

# ------------------------------
# 免費遊戲 mini slot
# ------------------------------
def draw_free_game_mini_slot(rng, bet_amount, multiplier):
    roll = torch.rand(1, generator=rng).item()
    if roll < 0.002312:
        return (bet_amount * 100) * multiplier
    elif roll < 0.07612:
        return (bet_amount * 50) * multiplier
    elif roll < 0.18612:
        return bet_amount * 25 * multiplier
    elif roll < 0.35112:
        return bet_amount * 10 * multiplier
    elif roll < 0.63512:
        return bet_amount * 5 * multiplier
    else:
        return bet_amount * 1 * multiplier

# ------------------------------
# 模擬單一玩家
# ------------------------------
def simulate_player(player_id):
    rng = torch.Generator().manual_seed(100 + player_id)
    pos_draw = torch.multinomial(probs, rounds_per_player, replacement=True, generator=rng)
    mult_draw = torch.multinomial(ball_probs, rounds_per_player, replacement=True, generator=rng)
    multipliers = ball_multipliers[mult_draw]
    base_prizes = payout_table[pos_draw]

    cum_profit = 0
    trend = []

    for i in range(rounds_per_player):
        pos = pos_draw[i].item()
        mult = multipliers[i].item()
        base_win = base_prizes[i].item() * mult
        reward = draw_free_game_mini_slot(rng, bet_amount, mult) if pos in free_game_positions else 0
        profit = base_win + reward - bet_amount
        cum_profit += profit
        trend.append(cum_profit)

    return trend

# ------------------------------
# 主流程：模擬並畫圖
# ------------------------------
all_trends = pd.DataFrame()

for pid in range(1, num_players + 1):
    print(f"模擬 Player {pid} 中...")
    trend = simulate_player(pid)
    all_trends[f'Player_{pid}'] = trend

# 畫圖
all_trends.index.name = "Round"
all_trends.plot(figsize=(12, 6), title="Median_nojp_95.5_趨勢圖")
plt.xlabel("遊戲局數")
plt.ylabel("累積淨利（贏的金額 - 投注）")
plt.grid(True)
plt.tight_layout()
plt.show()
