import os
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Direct path setup pointing directly past the nested folder structure
BASE_DIR = r"C:\Users\epshi\Downloads\Intel_image"

# Notice the double 'seg_train\seg_train' to reach the actual category subfolders
TRAIN_DIR = os.path.join(BASE_DIR, "seg_train", "seg_train") 
TEST_DIR = os.path.join(BASE_DIR, "seg_test", "seg_test")

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

print("🔄 Initializing dataset parsing...")
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    class_names = train_ds.class_names
    print(f"✅ Target categories successfully discovered: {class_names}")
    print(f"✅ Number of classes: {len(class_names)} (Should be 6)")
except Exception as e:
    print(f"❌ Configuration error: {e}")
    exit()

# Explicit Pixel Normalization Pipeline Mapping
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

train_ds = train_ds.map(normalize_img).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(normalize_img).cache().prefetch(tf.data.AUTOTUNE)

# Standardized CNN setup
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # Outputs 6 distinct probability distributions
    layers.Dense(len(class_names), activation='softmax') 
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

print("🚀 Retraining model pipeline...")
model.fit(train_ds, validation_data=val_ds, epochs=5)

model_path = os.path.join(BASE_DIR, "my_model.h5")
model.save(model_path)
print(f"🎉 Complete! Model file output to: {model_path}")