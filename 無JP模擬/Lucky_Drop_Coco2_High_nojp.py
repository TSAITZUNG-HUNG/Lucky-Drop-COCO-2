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
    1000.0, 125.0, 0, 0, 2.2,
    1.1, 0.5, 0.3, 0.0, 0.3,
    0.5, 1.1, 2.2, 0, 0,
    125.0, 1000.0
], dtype=torch.float32)

ball_multipliers = torch.tensor([1, 2, 5], dtype=torch.float32)
ball_probs = torch.tensor([0.93, 0.05, 0.02], dtype=torch.float32)

free_game_positions = {2, 3, 13, 14}
total_rounds = 100_000_000
batch_size = 10_000
save_interval = 20_000_000
bet_amount = 1.0
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------
# 免費遊戲 mini slot
# ------------------------------
def draw_free_game_mini_slot(rng, bet_amount, multiplier):
    roll = torch.rand(1, device=device, generator=rng).cpu().item()
    if roll < 0.00255:
        return (bet_amount * 200) * multiplier, "Super JP / X 200 獎", 0
    elif roll < 0.01275:
        return (bet_amount * 100) * multiplier, "Lucky JP / X 100 獎", 0
    elif roll < 0.07785:
        return bet_amount * 50 * multiplier, "X 50 獎", 0
    elif roll < 0.20396:
        return bet_amount * 25 * multiplier, "X 25 獎", 0
    elif roll < 0.49169:
        return bet_amount * 10 * multiplier, "X 10 獎", 0
    else:
        return bet_amount * 3 * multiplier, "X 3 獎", 0


# ------------------------------
# 單批模擬
# ------------------------------
def simulate_batch(rng, batch_size):
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
        pos = positions_drawn[i].item()
        mult = multipliers[i].item()
        base_win = base_prizes[i].item() * mult
        reward = label = 0

        if pos in free_game_positions:
            reward, label, _ = draw_free_game_mini_slot(rng, bet_amount, mult)

        total = base_win + reward
        records.append({
            'position': pos,
            'multiplier': mult,
            'base_game': base_win,
            'free_game_win': reward,
            'free_game_label': label if label else None,
            'total_payout': total
        })

    return records


# ------------------------------
# 分批執行模擬與分檔儲存
# ------------------------------
rng = torch.Generator(device=device).manual_seed(42)
part_id = 1
current_records = []

for i in range(total_rounds // batch_size):
    batch_records = simulate_batch(rng, batch_size)
    current_records.extend(batch_records)

    total_so_far = (i + 1) * batch_size
    print(f"[{total_so_far:,}] 筆模擬完成...")

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

        filename = f"lucky_drop_Coco2_High_nojp_95.5_part{part_id}.xlsx"
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            position_group.to_excel(writer, sheet_name="PositionStats", index=False)
            free_game_group.to_excel(writer, sheet_name="FreeGameRewards", index=False)

        print(f"✅ 已儲存 {filename}")
        current_records = []
        part_id += 1
