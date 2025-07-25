import torch
import pandas as pd

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
total_rounds = 1_000_000
batch_size = 5000
save_interval = 1_000_000
bet_amount = 1.0
jackpot_contribution_rate = 0.01
max_payout = 1.0

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ------------------------------
# 免費遊戲 mini slot with Jackpot
# ------------------------------
def draw_free_game_mini_slot(rng, jackpot_pool, bet_amount, multiplier):
    roll = torch.rand(1, device=device, generator=rng).cpu().item()
    if roll < 0.02312:
        jp_share = 0.25 * jackpot_pool * (bet_amount / max_payout)
        reward = (jp_share + bet_amount * 100) * multiplier
        label = "Super JP / X 100 獎"
    elif roll < 0.07527:
        jp_share = 0.05 * jackpot_pool * (bet_amount / max_payout)
        reward = (jp_share + bet_amount * 50) * multiplier
        label = "Lucky JP / X 50 獎"
    elif roll < 0.17532:
        reward = bet_amount * 25 * multiplier
        label = "X 25 獎"
        jp_share = 0.0
    elif roll < 0.31982:
        reward = bet_amount * 10 * multiplier
        label = "X 10 獎"
        jp_share = 0.0
    elif roll < 0.61347:
        reward = bet_amount * 5 * multiplier
        label = "X 5 獎"
        jp_share = 0.0
    else:
        reward = bet_amount * 1 * multiplier
        label = "X 1 獎"
        jp_share = 0.0
    return reward, label, jp_share

# ------------------------------
# 單批模擬
# ------------------------------
def simulate_batch(rng, batch_size, jackpot_pool):
    probs_gpu = probs.to(device)
    payout_table_gpu = payout_table.to(device)
    ball_probs_gpu = ball_probs.to(device)
    multipliers_gpu = ball_multipliers.to(device)

    positions_drawn = torch.multinomial(probs_gpu, num_samples=batch_size, replacement=True, generator=rng)
    multiplier_drawn = torch.multinomial(ball_probs_gpu, num_samples=batch_size, replacement=True, generator=rng)
    multipliers = multipliers_gpu[multiplier_drawn]
    base_prizes = payout_table_gpu[positions_drawn]

    records = []

    for i in range(batch_size):
        jackpot_pool += bet_amount * jackpot_contribution_rate
        pos = positions_drawn[i].item()
        mult = multipliers[i].item()
        base_win = base_prizes[i].item() * mult
        reward = 0.0
        label = None
        jp_share = 0.0

        if pos in free_game_positions:
            reward, label, jp_share = draw_free_game_mini_slot(rng, jackpot_pool, bet_amount, mult)
            jackpot_pool -= jp_share

        total = base_win + reward
        records.append({
            'position': pos,
            'multiplier': mult,
            'base_game': base_win,
            'free_game_win': reward,
            'free_game_label': label,
            'total_payout': total,
            'jackpot_pool': jackpot_pool
        })

    return records, jackpot_pool

# ------------------------------
# 主模擬與分檔儲存
# ------------------------------
rng = torch.Generator(device=device).manual_seed(42)
part_id = 1
current_records = []
jackpot_pool = 0.0

for i in range(total_rounds // batch_size):
    batch_records, jackpot_pool = simulate_batch(rng, batch_size, jackpot_pool)
    current_records.extend(batch_records)

    total_so_far = (i + 1) * batch_size
    if total_so_far % 10_000 == 0:
        print(f"[{total_so_far:,}] 筆完成，JP pool: {jackpot_pool:.2f}")

    if total_so_far % save_interval == 0:
        df = pd.DataFrame(current_records)

        # 匯出統計
        position_group = df.groupby('position').agg(
            count=('position', 'count'),
            total_win=('base_game', 'sum')
        ).reset_index()
        position_group['percentage'] = (position_group['count'] / len(df) * 100).round(4)

        free_game_group = df[df['free_game_label'].notna()].groupby('free_game_label').agg(
            count=('free_game_win', 'count'),
            total_win=('free_game_win', 'sum')
        ).reset_index()
        free_game_group['percentage'] = (free_game_group['count'] / len(df) * 100).round(4)

        filename = f"lucky_drop_Coco2_Medium_95.5_part{part_id}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            position_group.to_excel(writer, sheet_name="PositionStats", index=False)
            free_game_group.to_excel(writer, sheet_name="FreeGameRewards", index=False)
            #df.head(100_000).to_excel(writer, sheet_name="SampleDetails", index=False)

        print(f"✅ 已儲存 {filename}")
        current_records = []
        part_id += 1
