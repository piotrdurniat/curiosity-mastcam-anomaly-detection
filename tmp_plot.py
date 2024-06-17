import matplotlib.pyplot as plt
import os
import numpy as np

""" REMOVFE THIS FILE AFTER USING"""

train_dir = "./dataset/test_novel/all"


data = np.load(train_dir+"/"+'mcam00883_R0_sol0164_5.npy')
print(data.shape)

data = (data - np.min(data)) / (np.max(data) - np.min(data))

fig, axes = plt.subplots(1, 6, figsize=(20, 4))
for i in range(6):
    axes[i].imshow(data[:, :, i], cmap='gray')
    axes[i].set_title(f'Channel {i+1}')
    axes[i].axis('off')
    axes[i].imshow(data[:, :, i], cmap='gray')
    axes[i].set_title(f'Channel {i+1}')
    axes[i].axis('off')

plt.show()
rgb_image = np.take(data, [2,0,1], axis=2)
# Plot the combined RGB image
plt.imshow(rgb_image)

plt.title('Combined RGB Image')
plt.axis('off')
plt.show()