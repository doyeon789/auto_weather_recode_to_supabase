# 🌤 일사량 예측 및 전력 소비량 계산 시스템

# 1️⃣ Supabase 연동
from supabase import create_client
supabase = create_client("https://vcqqokmyyjsvxyvuzgmv.supabase.co", "API_KEY")

# 2️⃣ 데이터 불러오기 및 전처리
import pandas as pd, torch, torch.nn as nn, torch.optim as optim
from datetime import timedelta, datetime
past = supabase.table("r_weather_data").select("*").execute()
df = pd.DataFrame(past.data)
df['datetime'] = pd.to_datetime(df['r_timestamp'])
df['target'] = df['r_insolation'].replace(-9, 0)

# 3️⃣ LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self): 
        super().__init__()
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.linear = nn.Linear(64, 1)
    def forward(self, x): 
        return self.linear(self.lstm(x)[0][:, -1, :])

# 4️⃣ 학습
X = torch.tensor(df['target'].values[:-1]).view(-1, 13, 1)
y = torch.tensor(df['target'].values[1:]).view(-1, 1)
model = LSTMModel(); opt = optim.Adam(model.parameters(), lr=0.001)
for e in range(50):
    opt.zero_grad(); loss = ((model(X)-y)**2).mean(); loss.backward(); opt.step()

# 5️⃣ 24시간 예측
predictions = []
for h in range(24):
    pred = max(model(X[-1:]).item(), 0)
    if h < 6 or h > 18: pred = 0  # 밤 시간대 0 처리
    predictions.append((h, pred))

# 6️⃣ 전력 소비량 계산
def calc_power(I):
    return 1.0 + (5*0.2)*(I/1000) + (4*0.05) + (2*0.1) + 0.3

# 7️⃣ 결과 업로드
supabase.table("prediction").delete().execute()
records = [{"hour": h, "insolation": i, "power": calc_power(i)} for h, i in predictions]
supabase.table("prediction").insert(records).execute()

print("✅ 24시간 예측 및 전력 계산 완료")