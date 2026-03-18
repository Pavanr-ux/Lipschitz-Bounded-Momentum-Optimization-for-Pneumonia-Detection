import matplotlib.pyplot as plt

custom = [2.0, 1.4, 1.0]
adam = [2.2, 1.8, 1.3]

plt.plot(custom, label="Custom")
plt.plot(adam, label="Adam")

plt.legend()
plt.title("Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.show()