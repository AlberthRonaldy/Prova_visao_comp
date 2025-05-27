import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import cv2
import os
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

classes_to_use = [3, 5]

train_filter = np.isin(y_train, classes_to_use).flatten()
x_train_filtered = x_train[train_filter]
y_train_filtered = y_train[train_filter]

test_filter = np.isin(y_test, classes_to_use).flatten()
x_test_filtered = x_test[test_filter]
y_test_filtered = y_test[test_filter]

x_filtered = np.concatenate((x_train_filtered, x_test_filtered), axis=0)
y_filtered = np.concatenate((y_train_filtered, y_test_filtered), axis=0)

# Mapear rótulos para 0 (gato) e 1 (cachorro)
y_filtered = (y_filtered == classes_to_use[1]).astype(np.int32)

x_train_final, x_test_final, y_train_final, y_test_final = train_test_split(
    x_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # saída binária 0 ou 1
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_final, y_train_final, epochs=10, batch_size=64, validation_split=0.1)

y_pred_prob = model.predict(x_test_final)
y_pred = (y_pred_prob > 0.5).astype(int)

precision = precision_score(y_test_final, y_pred)
recall = recall_score(y_test_final, y_pred)
f1 = f1_score(y_test_final, y_pred)

print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Caminho das imagens
path = ".\imagens"
image_files = os.listdir(path)

# Armazenar imagens originais e pré-processadas
images_original = []
images_model_input = []

for img_file in image_files:
    img_path = os.path.join(path, img_file)

    img = cv2.imread(img_path)
    if img is None:
        print(f"Erro ao carregar: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images_original.append((img_rgb, img_file))

    img_resized = cv2.resize(img_rgb, (32, 32))

    # Normalizar (igual ao treino do modelo)
    img_normalized = img_resized.astype('float32') / 255.0

    # Adicionar para predição (expandir dimensão para batch)
    images_model_input.append(img_normalized)

# Converter para array
images_model_input = np.array(images_model_input)

# Fazer predições
predictions = model.predict(images_model_input)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Mostrar os resultados
for i, (img, filename) in enumerate(images_original):
    label = "Cachorro" if predicted_labels[i] == 1 else "Gato"
    plt.figure(figsize=(2, 2))
    plt.imshow(img)
    plt.title(f"{filename} → {label}")
    plt.axis('off')
    plt.show()
