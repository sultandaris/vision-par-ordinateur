import tensorflow as tf
from tensorflow import keras
# Refer to layers via the keras object to avoid editor/linter import resolution issues
layers = keras.layers

# 1. Muat & Preprocess Dataset CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalisasi data: ubah piksel dari rentang [0, 255] ke [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ubah label menjadi one-hot encoding
y_train_categorical = keras.utils.to_categorical(y_train, 10)
y_test_categorical = keras.utils.to_categorical(y_test, 10)

print(f"Bentuk data training asli (x_train): {x_train.shape}")
# Output: (50000, 32, 32, 3)

# 2. Tentukan Arsitektur Model LSTM
model = keras.Sequential()

# Tentukan input shape asli (32x32x3)
model.add(layers.Input(shape=(32, 32, 3)))

# --- LANGKAH PENTING ---
# Ubah bentuk (Reshape) data agar sesuai untuk LSTM
# (32, 32, 3) -> (32 timesteps, 96 features)
model.add(layers.Reshape(target_shape=(32, 96)))

# Tambahkan layer LSTM dengan 128 unit
model.add(layers.LSTM(128))

# Tambahkan layer output
# 10 unit untuk 10 kelas
model.add(layers.Dense(10, activation="softmax"))

# Tampilkan ringkasan model
model.summary()

# 3. Compile Model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# 4. Latih Model
print("\n--- Memulai Pelatihan Model ---")
batch_size = 64
epochs = 10 # Kita butuh lebih banyak epoch untuk CIFAR-10

history = model.fit(
    x_train,
    y_train_categorical,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)
print("--- Pelatihan Selesai ---")

# 5. Evaluasi Model dengan Data Test
print("\n--- Mengevaluasi Model dengan Data Test ---")
score = model.evaluate(x_test, y_test_categorical, verbose=0)
print(f"Test loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]:.4f}")