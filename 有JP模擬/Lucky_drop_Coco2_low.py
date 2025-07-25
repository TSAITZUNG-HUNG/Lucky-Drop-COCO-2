import numpy as np
import pandas as pd

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
rounds = 100_000_000
bet_amount = 1

# ------------------------------
# 模擬主程式
# ------------------------------
def simulate_lucky_drop_no_free_game(rounds=rounds, seed=42):
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(rounds):
        pos = rng.choice(positions, p=probs)
        base_prize = payout_table[pos]
        multiplier = rng.choice(ball_multipliers, p=ball_probs)
        payout = base_prize * multiplier

        records.append({
            'position': pos,
            'multiplier': multiplier,
            'base_game': base_prize,
            'total_payout': payout
        })

    df = pd.DataFrame(records)

    summary = {
        'RTP_total': df['total_payout'].mean(),
        'final_rounds': rounds
    }

    return df, summary

# ------------------------------
# 執行模擬與統計分析
# ------------------------------
df, summary = simulate_lucky_drop_no_free_game(rounds=rounds)

# 各位置出現統計
position_stats = df['position'].value_counts().sort_index()
position_pct = (position_stats / len(df) * 100).round(4)
position_df = pd.DataFrame({
    'position': position_stats.index,
    'count': position_stats.values,
    'percentage': position_pct.values
})

# 倍率球統計
multiplier_stats = df['multiplier'].value_counts().sort_index()
multiplier_pct = (multiplier_stats / len(df) * 100).round(4)
multiplier_df = pd.DataFrame({
    'multiplier': multiplier_stats.index,
    'count': multiplier_stats.values,
    'percentage': multiplier_pct.values
})

# ------------------------------
# 儲存成 Excel（多工作表）
# ------------------------------
with pd.ExcelWriter("lucky_drop_no_free_game.xlsx", engine='openpyxl') as writer:
    position_df.to_excel(writer, sheet_name="PositionStats", index=False)
    multiplier_df.to_excel(writer, sheet_name="BallMultipliers", index=False)
    # df.to_excel(writer, sheet_name="SimulationDetails", index=False)

# ------------------------------
# 印出 RTP 統計摘要
# ------------------------------
print("🎯 模擬結果摘要")
print(f"整體 RTP（不含免費遊戲） ：{summary['RTP_total']:.6f}")
print(f"模擬總局數                  ：{summary['final_rounds']:,}")
