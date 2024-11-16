from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from sklearn.metrics import classification_report


# Đường dẫn tới dữ liệu huấn luyện và kiểm tra
train_dir = 'data/train'
test_dir = 'data/test'
num_classes = 4  # Số loại phương tiện: xe hơi, xe tải, xe máy, xe buýt

# Load mô hình VGG16 đã huấn luyện trước
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Thêm các layer cuối cùng cho mô hình
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Đóng băng các layer trong mô hình gốc
for layer in base_model.layers:
    layer.trainable = False

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation để tạo dữ liệu đa dạng hơn
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# Huấn luyện mô hình
model.fit(train_data, epochs=10, validation_data=test_data)

# Lưu mô hình sau khi huấn luyện
model.save('vehicle_classification_model.keras')

# Dự đoán trên tập kiểm tra
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_data.classes

# In báo cáo kết quả
print(classification_report(y_true, y_pred_classes, target_names=list(train_data.class_indices.keys())))

