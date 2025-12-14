import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/Train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/Val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/Test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# 1. Improved Data Augmentation (Slightly more aggressive)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2), # Increased from 0.1
    tf.keras.layers.RandomZoom(0.2), # Increased from 0.1
    tf.keras.layers.RandomContrast(0.2) # Added contrast adjustment
])

class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO

# Apply options and the new ignore_errors() method directly to the datasets
train_ds = train_ds.with_options(options).ignore_errors()
val_ds = val_ds.with_options(options).ignore_errors()
test_ds = test_ds.with_options(options).ignore_errors()

# Cache & prefetch datasets
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds  = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization = tf.keras.layers.Rescaling(1./255)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# --- STAGE 1: Feature Extraction (Training Custom Head) ---

base_model.trainable = False # Freeze the base model

model = tf.keras.Sequential([
    data_augmentation,
    normalization,
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2), # 2. Added Dropout layer to fight overfitting
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam", # Standard learning rate is fine for this stage
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

print("\n--- Starting Stage 1: Feature Extraction (Frozen Base) ---")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# --- STAGE 2: Fine-Tuning (Unfreezing the Base Model) ---

# 3. Unfreeze the base model
base_model.trainable = True

# 4. Recompile with a very low learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # CRITICAL for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Set a much longer epoch count for fine-tuning
fine_tune_epochs = 20
total_epochs = history.epoch[-1] + fine_tune_epochs

print("\n--- Starting Stage 2: Fine-Tuning (Unfrozen Base with Low LR) ---")
history_fine_tune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Start from the epoch where Stage 1 left off
    callbacks=[early_stop]
)

# --- Evaluation and TFLite Conversion ---
# The model has been restored to the best weights from either stage

test_loss, test_acc = model.evaluate(test_ds)
print("\nFinal Test accuracy after Fine-Tuning:", test_acc)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open("hand_gesture.tflite", "wb") as f:
    f.write(tflite_model)