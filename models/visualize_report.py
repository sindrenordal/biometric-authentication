import pandas as pd
import matplotlib.pyplot as plt

from LSTM import load_dataframe

# Load data
user1 = load_dataframe(["100669"])
user2 = load_dataframe(["180679"])

print("user1 shape", user1.shape)
print("user2 shape", user2.shape)

# Keep first window
user1_window = user1[0,:,:]
user2_window = user2[0,:,:]

# Prepare data
user1_x = user1[0,:,0]
user1_y = user1[0,:,1]
user1_z = user1[0,:,2]

user2_x = user2[0,:,0]
user2_y = user2[0,:,1]
user2_z = user2[0,:,2]

# Plot data
fig = plt.figure()

ax1 = fig.add_subplot(131)
ax1.plot(user1_x, color='r', label="user1_x")
ax1.plot(user2_x, color='b', label="user2_x")
ax1.legend()

ax2 = fig.add_subplot(132)
ax2.plot(user1_y, color='r', label="user1_y")
ax2.plot(user2_y, color='b', label="user2_y")
ax2.legend()

ax3 = fig.add_subplot(133)
ax3.plot(user1_z, color='r', label="user1_z")
ax3.plot(user2_z, color='b', label="user2_z")
ax3.legend()

plt.show()



