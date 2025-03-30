import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PIL import Image

MODEL_PATH = 'gun_classify_model24.keras'


class CustomIterator(DirectoryIterator):
    def __init__(self, *args, **kwargs):
        self.invalid_images = kwargs.pop('invalid_images', [])
        super().__init__(*args, **kwargs)
        self.valid_indices = [i for i, filepath in enumerate(self.filepaths) if filepath not in self.invalid_images]
        self.samples = len(self.valid_indices)

    def __len__(self):
        return int(np.ceil(self.samples / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_indices = self.valid_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self._get_batches_of_transformed_samples([i])[0][0] for i in batch_indices])
        batch_y = np.array([self.classes[i] for i in batch_indices])
        return batch_x, batch_y

    def get_valid_classes(self):
        return np.array([self.classes[i] for i in self.valid_indices])


class DynamicClassWeightCallback(Callback):
    def __init__(self, val_data, update_frequency=3, target_accuracy=0.8, min_weight=0.5, max_weight=2.0):
        super().__init__()
        self.val_data = val_data
        self.update_frequency = update_frequency
        self.class_weights = {0: 1.0, 1: 1.0}
        self.target_accuracy = target_accuracy
        self.min_weight = min_weight
        self.max_weight = max_weight

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.update_frequency == 0:
            predictions = self.model.predict(self.val_data)
            predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
            true_classes = self.val_data.get_valid_classes()

            # 确保长度匹配
            min_length = min(len(predicted_classes), len(true_classes))
            predicted_classes = predicted_classes[:min_length]
            true_classes = true_classes[:min_length]

            class_accuracies = []
            for i in range(2):
                mask = (true_classes == i)
                class_acc = np.mean(predicted_classes[mask] == true_classes[mask])
                class_accuracies.append(class_acc)

            if all(acc >= self.target_accuracy for acc in class_accuracies):
                print(f"\nEpoch {epoch + 1} - Both classes reached target accuracy, stopping weight adjustment")
                return

            total_acc = sum(class_accuracies)
            new_weights = [total_acc / (2 * acc) for acc in class_accuracies]
            new_weights = np.clip(new_weights, self.min_weight, self.max_weight)
            new_weights = new_weights / np.sum(new_weights) * 2

            self.class_weights = {0: new_weights[0], 1: new_weights[1]}
            print(f"\nEpoch {epoch + 1} - Updating class weights: {self.class_weights}")
            print(f"Class accuracies - Class 0: {class_accuracies[0]:.4f}, Class 1: {class_accuracies[1]:.4f}")

    def on_batch_begin(self, batch, logs=None):
        self.model.class_weight = self.class_weights


def find_invalid_images(directory):
    invalid_images = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError, Image.UnidentifiedImageError):
                print(f"Invalid image detected: {file_path}")
                invalid_images.append(file_path)
    return invalid_images


def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def custom_flow_from_directory(directory, invalid_images, *args, **kwargs):
    datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=180,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    return CustomIterator(
        directory,
        datagen,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        invalid_images=invalid_images,
        *args,
        **kwargs
    )


def train_model():
    model = create_model()

    train_generator = custom_flow_from_directory(
        'E:/机器学习训练数据集/训练集',
        invalid_train_images,
        shuffle=True
    )

    val_generator = custom_flow_from_directory(
        'E:/机器学习训练数据集/验证集',
        invalid_val_images,
        shuffle=False
    )

    dynamic_weight_callback = DynamicClassWeightCallback(
        val_generator,
        update_frequency=4,
        target_accuracy=0.9,
        min_weight=0.8,
        max_weight=1.2
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=5, mode='max', restore_best_weights=True),
        dynamic_weight_callback,
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
    ]

    # Initial training
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=callbacks
    )

    # Fine-tuning
    for layer in model.layers[-5:]:
        layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    fine_tune_history = model.fit(
        train_generator,
        epochs=40,
        validation_data=val_generator,
        callbacks=callbacks
    )

    print(f"Model saved as '{MODEL_PATH}'")
    return history, fine_tune_history


def evaluate_model(model, test_generator, best_threshold):
    predictions = model.predict(test_generator)
    predicted_classes = (predictions > best_threshold).astype(int).reshape(-1)
    true_classes = test_generator.get_valid_classes()

    # 确保长度匹配
    min_length = min(len(predicted_classes), len(true_classes))
    predicted_classes = predicted_classes[:min_length]
    true_classes = true_classes[:min_length]

    class_labels = list(test_generator.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_classes, predicted_classes))

    accuracy = accuracy_score(true_classes, predicted_classes)
    print(f"\nOverall accuracy: {accuracy:.4f}")

    class_0_acc = accuracy_score(true_classes[true_classes == 0], predicted_classes[true_classes == 0])
    class_1_acc = accuracy_score(true_classes[true_classes == 1], predicted_classes[true_classes == 1])
    print(f"Class 0 accuracy: {class_0_acc:.4f}")
    print(f"Class 1 accuracy: {class_1_acc:.4f}")

    for i in range(min_length):
        print(f"Image path: {test_generator.filepaths[test_generator.valid_indices[i]]}")
        print(f"True label: {class_labels[true_classes[i]]}")
        print(f"Predicted label: {class_labels[predicted_classes[i]]}")
        print(f"Prediction probability: {predictions[i][0]:.4f}")
        print("-" * 30)


def plot_training_history(history, fine_tune_history):
    def plot_metric(history, metric, title):
        plt.figure(figsize=(10, 4))
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(title)
        plt.ylabel(metric)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    plot_metric(history, 'accuracy', 'Model Accuracy (Initial Training)')
    plot_metric(history, 'loss', 'Model Loss (Initial Training)')
    plot_metric(fine_tune_history, 'accuracy', 'Model Accuracy (Fine Tuning)')
    plot_metric(fine_tune_history, 'loss', 'Model Loss (Fine Tuning)')


def find_best_threshold(model, val_generator):
    predictions = model.predict(val_generator)
    true_classes = val_generator.get_valid_classes()

    # 确保长度匹配
    min_length = min(len(predictions), len(true_classes))
    predictions = predictions[:min_length]
    true_classes = true_classes[:min_length]

    thresholds = np.arange(0.1, 1.0, 0.05)
    accuracies = []

    for threshold in thresholds:
        predicted_classes = (predictions > threshold).astype(int).reshape(-1)
        accuracy = accuracy_score(true_classes, predicted_classes)
        accuracies.append(accuracy)

    best_threshold = thresholds[np.argmax(accuracies)]
    print(f"Best threshold: {best_threshold:.2f}")
    return best_threshold


if __name__ == "__main__":
    # 查找无效图像
    invalid_train_images = find_invalid_images('E:/机器学习训练数据集/训练集')
    invalid_val_images = find_invalid_images('E:/机器学习训练数据集/验证集')
    invalid_test_images = find_invalid_images('E:/机器学习训练数据集/测试集')
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")

        model = tf.keras.models.load_model(MODEL_PATH)
    else:



        print("Model does not exist, starting training...")
        history, fine_tune_history = train_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        plot_training_history(history, fine_tune_history)

    val_generator = custom_flow_from_directory(
        'E:/机器学习训练数据集/验证集',
        invalid_val_images,
        shuffle=False
    )

    best_threshold = find_best_threshold(model, val_generator)

    test_generator = custom_flow_from_directory(
        'E:/机器学习训练数据集/测试集',
        invalid_test_images,
        shuffle=False
    )

    evaluate_model(model, test_generator, best_threshold)

