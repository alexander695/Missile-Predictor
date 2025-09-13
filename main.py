import numpy as np
import matplotlib.pyplot as plt

# Trayectoria real con thruster
g = 9.81
v0 = 300
angle = np.radians(60)
t = np.linspace(0, 60, 200)

# Balistica
x_true = v0 * np.cos(angle) * t
y_true = v0 * np.sin(angle) * t - 0.5 * g * t**2

# Thruster
T = 200 * np.sin(0.1 * t) * np.exp(-0.02 * t)
y_thruster = y_true + T

mask = y_thruster >= 0
x_true, y_thruster, t = x_true[mask], y_thruster[mask], t[mask]

# Normalizar
x_norm = x_true / np.max(x_true)
y_norm = y_thruster / np.max(y_thruster)

# Dataset (ventana)
window = 4
X, Y = [], []
for i in range(window, len(x_norm)):
    X.append(np.hstack([x_norm[i-window:i], y_norm[i-window:i]]))
    Y.append([x_norm[i], y_norm[i]])
X, Y = np.array(X), np.array(Y)

# Red neuronal basica con NumPy
def init_network(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    W1 = np.random.randn(input_dim, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    W2 = np.random.randn(hidden_dim, output_dim) * 0.1
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    h = np.tanh(X @ W1 + b1)
    out = h @ W2 + b2
    return h, out

# Funcion de perdida con barrera logaritmica

def loss_with_barrier(Y, Y_pred, weights, M=1.0, mu=1e-3):
    mse = np.mean((Y - Y_pred)**2)
    barrier = 0
    for W in weights:
        barrier -= mu * np.sum(np.log(M - W) + np.log(M + W))
    return mse + barrier

# Entrenamiento con Newton-Barrier simplificado

W1, b1, W2, b2 = init_network(X.shape[1], 10, 2)
lr = 0.05
M = 1.0
mu = 1e-3
epochs = 500

for epoch in range(epochs):
    # Forward
    h, Y_pred = forward(X, W1, b1, W2, b2)

    # Loss
    loss = loss_with_barrier(Y, Y_pred, [W1, W2], M, mu)

    # Gradientes (retroprop simplificada)
    dY = (Y_pred - Y) / len(Y)
    dW2 = h.T @ dY
    db2 = np.sum(dY, axis=0, keepdims=True)

    dh = dY @ W2.T * (1 - h**2)
    dW1 = X.T @ dh
    db1 = np.sum(dh, axis=0, keepdims=True)

    # Actualizacion (descenso + barrera)
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Clipping preventivo (para no salir del rango)
    W1 = np.clip(W1, -M*0.99, M*0.99)
    W2 = np.clip(W2, -M*0.99, M*0.99)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Prediccion
_, pred = forward(X, W1, b1, W2, b2)

plt.figure(figsize=(10,7))
plt.plot(x_norm, y_norm, 'k-', label="Trayectoria real (thruster)")
plt.plot(X[:, -2], X[:, -1], 'o', alpha=0.3, label="Radar (medido)")
plt.plot(pred[:,0], pred[:,1], 'x--', label="Predicción NN (Newton-Barrier)")
plt.xlabel("x (normalizado)")
plt.ylabel("y (normalizado)")
plt.legend()
plt.title("Trayectoria con Newton-Barrier (restricciones en pesos)")
plt.show()
