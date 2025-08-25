import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_spiral(N=100, noise=0.2):
    theta = np.sqrt(np.random.rand(N)) * 2 * np.pi  
    r_a = 2*theta + np.pi
    data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
    x_a = data_a + np.random.randn(N,2) * noise
    y_a = np.zeros(N)

    theta = np.sqrt(np.random.rand(N)) * 2 * np.pi
    r_b = -2*theta - np.pi
    data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
    x_b = data_b + np.random.randn(N,2) * noise
    y_b = np.ones(N)

    X = np.vstack([x_a, x_b])
    y = np.hstack([y_a, y_b])
    return X, y

X, y = generate_spiral(N=500)

df = pd.DataFrame(X, columns=['x', 'y'])
df['label'] = y

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

train_df.to_csv('spiral_train.csv', index=False)
test_df.to_csv('spiral_test.csv', index=False)