import requests
import pandas as pd
from datetime import datetime, timedelta
import pytz  # pytz 임포트

from supabase import create_client, Client

url = "https://vcqqokmyyjsvxyvuzgmv.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZjcXFva215eWpzdnh5dnV6Z212Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA5MjE2OTgsImV4cCI6MjA3NjQ5NzY5OH0.lv0mtev8N61_QicEObv5Bdbk7Gpwnh-tLnkX0M-SI5Q"
supabase: Client = create_client(url, key)

API_KEY = "Ci_fVsYCSNKv31bGAijSTA"
STN = 108  # 서울

# pytz로 한국 시간대 설정
kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)

# 기준 시각: 한국 시간 기준 현재 시각에서 분/초/마이크로초를 0으로 맞춤 (마지막 정시)
END_TIME = now.replace(minute=0, second=0, microsecond=0)
START_TIME = END_TIME  # 단일 시각 요청용
DELTA = timedelta(minutes=3)  # (사용 안됨)

data = {
    "r_timestamp": None,
    "r_temperature": None,
    "r_humidity": None,
    "r_insolation": None
}

print(f"📅 기준 시각: {END_TIME.strftime('%Y-%m-%d %H:%M')} (마지막 정시, KST)")

def fetch_data_for_time(target_time):
    TM = target_time.strftime("%Y%m%d%H%M")
    URL = f"https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php?tm={TM}&stn={STN}&authKey={API_KEY}"
    print(f"⏳ Requesting data for {TM} ...")

    try:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
        text = response.text.strip()

        # API 오류 체크
        if "ERROR" in text or "help" in text.lower():
            print(f"🚨 API returned error at {TM}, skipping.")
            return []

        lines = text.split("\n")
        data_lines = lines[2:]  # 헤더 2줄 스킵

        records = []
        for line in data_lines:
            parts = line.strip().split()
            if len(parts) < 36 or parts[0].startswith("#"):
                continue
            
            try:
                ts = datetime.strptime(parts[0], "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:00")
            except:
                ts = parts[0]

            record = {
                "timestamp": ts,
                "Temperature": float(parts[11]),
                "Humidity": float(parts[13]),
                "insolation": float(parts[34])
            }

            records.append(record)

        if records:
            print(f"✅ Data collected for {TM}")
        else:
            print(f"⚠️ No valid data in response for {TM}")
        return records

    except Exception as e:
        print(f"⚠️ Exception occurred at {TM}: {e}")
        return []

# 실제 데이터 요청 실행
records = fetch_data_for_time(END_TIME)

if records:
    first = records[0]
    data.update({
        "r_timestamp": first["timestamp"],
        "r_temperature": first["Temperature"],
        "r_humidity": first["Humidity"],
        "r_insolation": first["insolation"]
    })
else:
    print("❌ No data collected.")

response = supabase.table("r_weather_data").insert(data).execute()
print(response)
