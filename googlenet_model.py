# googleNet_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt


# 데이터셋 경로
TRAIN_DATASET_PATH = 'D:/kor_face_ai/kor_face/Training'  # 학습 데이터셋 경로
VALIDATION_DATASET_PATH = 'D:/kor_face_ai/kor_face/Validation'  # 검증 데이터셋 경로

# 데이터셋 불러오기
def load_datasets(batch_size=32, img_size=(299, 299)):
    train_dataset = image_dataset_from_directory(
        TRAIN_DATASET_PATH,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())

    
    validation_dataset = image_dataset_from_directory(
        TRAIN_DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=img_size,
        batch_size=batch_size
    )

    # 오류 무시 추가
    validation_dataset = validation_dataset.apply(tf.data.experimental.ignore_errors())
    
    return train_dataset, validation_dataset

# GoogleNet (Inception-V3) 모델 정의
def build_googlenet_model():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    # 하이퍼파라미터 설정
    BATCH_SIZE = 16
    IMG_SIZE = (299, 299)

    # 데이터셋 로드
    train_dataset, validation_dataset = load_datasets(batch_size=BATCH_SIZE, img_size=IMG_SIZE)

    # 모델 생성
    model = build_googlenet_model()

    # 모델 컴파일
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

    # 학습 및 검증 정확도 그래프 그리기
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
