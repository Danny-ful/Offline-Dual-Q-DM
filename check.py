import pickle

with open("experts/HalfCheetah-v2_d4rl.pkl", "rb") as f:
    data = pickle.load(f)

print("数据字典里的键有:", list(data.keys()))

# 看看 rewards 到底是个啥
if 'rewards' in data:
    print("前20步奖励样例:", data['rewards'][:20])
elif 'reward' in data:
    print("前20步奖励样例:", data['reward'][:20])