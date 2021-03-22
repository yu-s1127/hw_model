# %% [markdown]
# # simple model

# %%
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# %% [markdown]
# # load data

# %%
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %%
class_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# %%
data = {
    "train_images": train_images,
    "test_images": test_images,
    "train_labels": train_labels,
    "test_labels": test_labels
}

# %% [markdown]
# ## type

# %%
for key in data:
    print(f"{key}: {type(data[key])}")

# %% [markdown]
# ## shape

# %%
for key in data:
    print(f"{key}: {data[key].shape}")

# %% [markdown]
# ## range

# %%
for key in data:
    print(f"{key}: {data[key].max()},{data[key].min()} ")

# %% [markdown]
# ## labels

# %%
np.unique(train_labels), np.unique(test_labels)

# %% [markdown]
# ## sample

# %%
train_images[0,:,:]

# %% [markdown]
# ## look at data

# %%
plt.figure()
# plt.imshow(train_images[0])
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

# %%
train_labels[0]

# %%
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[train_labels[i]])

plt.show()

# %% [markdown]
# # preprocess data

# %%
train_images_norm = train_images / 255.0
test_images_norm = test_images /255.0

# %%
train_images_norm.min(), train_images_norm.max()

# %%
train_images_norm.min(), train_images_norm.max()

# %%
plt.figure()
# plt.imshow(train_images[0])
plt.imshow(train_images_norm[0], cmap=plt.cm.binary)
plt.colorbar()
plt.show()

# %%
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images_norm[i], cmap=plt.cm.binary)
    plt.xlabel(class_name[train_labels[i]])

plt.show()

# %% [markdown]
# # build the model

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# %%
def relu(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
plt.ylim(-1, 5)
plt.grid()
plt.plot(x, relu(x))
plt.show()

# %%
model.summary()

# %% [markdown]
# # compile the model

# %%
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer="adam",
    metrics=["accuracy"]
)

# %% [markdown]
# # fit

# %%
model.fit(
    train_images_norm,
    train_labels,
    epochs=5
)

# %%
60000 / 32

# %% [markdown]
# # evaluate

# %%
model.evaluate(test_images_norm, test_labels, verbose=2)

# %%
10000 / 32

# %% [markdown]
# # predict

# %%
raw_predictions = model.predict(test_images_norm)

# %%
raw_predictions.shape

# %%
raw_predictions[0]

# %%
np.argmax(raw_predictions[0])

# %%
test_labels[0]

# %% [markdown]
# ## prob model

# %% [markdown]
# prob_model = tf.keras.Sequential([
#     model,
#     tf.keras.layers.Softmax()
# ])

# %%
predictions = prob_model.predict(test_images_norm)

# %%
predictions.shape

# %%
predictions[0]

# %%
predictions[0].sum()

# %%
np.argmax(predictions[0])

# %%
np.argsort(-predictions[0])

# %%
np.argsort(-raw_predictions[0])

# %% [markdown]
# # verify model

# %%
plt.imshow(test_images_norm[0], cmap=plt.cm.binary)
plt.show()

# %%
plt.bar(range(10), predictions[0])
plt.xticks(range(10))
plt.show()

# %%
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(test_images_norm[0], cmap=plt.cm.binary)

plt.subplot(1,2,2)
plt.bar(range(10), predictions[0])
plt.xticks(range(10))

plt.show()


# %%
def verify_prediction(i):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(test_images_norm[i], cmap=plt.cm.binary)

    plt.subplot(1,2,2)
    plt.bar(range(10), predictions[i])
    plt.xticks(range(10))

    plt.show()


# %%
verify_prediction(1)

# %%
verify_prediction(2)

# %%
