import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import yfinance as yf

class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.n_steps = len(df)
        self.current_step = 0
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.net_worths = []
        self.daily_returns = []
        self.volatility_target = 0.02  # # volatility 2%
        self.previous_volatility = None
        self.dates = []  

    def reset(self):
        self.current_step = np.random.randint(0, len(self.df) - 1)  
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.net_worths = []
        self.daily_returns = []
        self.previous_volatility = None
        self.dates = []  # 重置日期信息
        return self._next_observation()

    def _next_observation(self):
        obs = np.array([
            self.df.iloc[self.current_step]['Open'],
            self.df.iloc[self.current_step]['High'],
            self.df.iloc[self.current_step]['Low'],
            self.df.iloc[self.current_step]['Close'],
            self.df.iloc[self.current_step]['Volume'],
            self.df.iloc[self.current_step]['MACD'],
            self.df.iloc[self.current_step]['RSI']
        ])
        return obs / np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0.001  

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price * (1 + transaction_cost)
            self.shares_held += shares_bought
        elif action == 2:  # Sell
            self.balance += self.shares_held * current_price * (1 - transaction_cost)
            self.shares_held = 0

        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worths.append(self.net_worth)
        self.dates.append(self.df.index[self.current_step]) 
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth != 0 else 0
        self.daily_returns.append(daily_return)

        if len(self.daily_returns) >= 60:
            self.previous_volatility = np.std(self.daily_returns[-60:])

        if self.previous_volatility:
            volatility_scaling = self.volatility_target / self.previous_volatility
        else:
            volatility_scaling = 1

        reward = (volatility_scaling * action * daily_return) - (volatility_scaling * transaction_cost * action)

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Total shares sold: {self.total_shares_sold}')
        print(f'Total sales value: {self.total_sales_value}')

    def get_net_worths(self):
        return self.net_worths

    def get_dates(self):
        return self.dates  
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(state_size, 64, 2, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = torch.relu(self.fc1(x[:, -1, :]))
        x = self.fc2(x)
        return x.squeeze(1)         
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(state_size, action_size).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values[0]).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)[0]).item()
            target_f = self.model(state)
            target_f = target_f.clone()
            target_f[0][action] = target  
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def fetch_stock_data(stock_ticker):
    df = yf.download(stock_ticker, start="2023-01-01", end="2024-06-01")
    df = add_technical_indicators(df)
    return df

def add_technical_indicators(df):
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df = df.dropna()
    return df

# load and test
if __name__ == "__main__":
    # load train model
    agent = DQNAgent(state_size, action_size)
    agent.load("/content/dqn_30.pth")  # load
    

    # test
    test_df = fetch_stock_data("VOO")  
    test_env = StockTradingEnv(test_df)
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.n

    
    state = test_env.reset()
    state = np.reshape(state, [1, state_size])
    test_net_worths = []
    test_dates = []

    
    while True:
        action = agent.act(state)  
        next_state, reward, done, _ = test_env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        test_net_worths.append(test_env.net_worth)
        test_dates.append(test_env.df.index[test_env.current_step])  
            break

    # draw test data
    plt.plot(test_dates, test_net_worths, label='Test Net Worth')
    plt.xlabel('Date')
    plt.ylabel('Net Worth')
    plt.title('Net Worth Over Time on Test Data')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()