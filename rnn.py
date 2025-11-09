import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from supabase import create_client

# --- 1. Supabase 데이터 불러오기 ---
url = "https://vcqqokmyyjsvxyvuzgmv.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZjcXFva215eWpzdnh5dnV6Z212Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5MjE2OTgsImV4cCI6MjA3NjQ5NzY5OH0.lv0mtev8N61_QicEObv5Bdbk7Gpwnh-tLnkX0M-SI5Q"
supabase = create_client(url, key)

past_resp = supabase.table("weather_data").select("*").execute()
past_data = pd.DataFrame(past_resp.data)
realtime_resp = supabase.table("r_weather_data").select("*").execute()
realtime_data = pd.DataFrame(realtime_resp.data)

# --- 2. 전처리 ---
past_data['datetime'] = pd.to_datetime(past_data['timestamp'])
realtime_data['datetime'] = pd.to_datetime(realtime_data['r_timestamp'])
past_data = past_data.rename(columns={'insolation': 'target'})
realtime_data = realtime_data.rename(columns={'r_insolation': 'target'})

past_data = past_data.dropna(subset=['target'])
realtime_data = realtime_data.dropna(subset=['target'])

# --- 3. 시계열 시퀀스 생성 ---
def create_sequences(data, seq_length=24):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data['target'].iloc[i:i+seq_length].values
        y = data['target'].iloc[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(past_data, seq_length=24)

# numpy -> torch tensor 변환
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # (batch, seq_len, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # (batch, 1)

# --- 4. LSTM 모델 정의 ---
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        out = out[:, -1, :]    # 마지막 타임스텝 출력만 사용
        out = self.linear(out) # (batch, 1)
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 5. 학습 ---
epochs = 50
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --- 6. 실시간 데이터로 예측 ---
realtime_data = realtime_data.sort_values('datetime').reset_index(drop=True)
recent_24h = realtime_data['target'].iloc[-24:].values
recent_24h = torch.tensor(recent_24h, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 24, 1)

model.eval()
with torch.no_grad():
    prediction = model(recent_24h)
print(f"다음 시간 예측 일사량: {prediction.item():.2f}")
