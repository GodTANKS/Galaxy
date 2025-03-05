import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time  # time 모듈을 추가
import requests
from io import BytesIO


def main():
    st.title("단계별 은하 분류 데이터 준비 및 분석")
    st.write("1단계: 데이터 수집 → 1-1단계: 데이터 수집 결과 요약 → 2-3단계: 데이터 처리 및 탐색 → 4단계: 데이터 분석 및 표현")

    BASE_FOLDER = "https://github.com/GodTANKS/Galaxy/raw/main"
    FOLDER_NAMES = {
        "0": "Elliptical Galaxy",
        "1": "Lens Galaxy",
        "2": "Spiral Galaxy",
        "3": "Barred Spiral Galaxy",
        "4": "Irregular Galaxy",
    }

    # 세션 상태 초기화
    if 'selected_folder' not in st.session_state:
        st.session_state.selected_folder = None
    if 'collected_data' not in st.session_state:
        st.session_state.collected_data = {key: [] for key in FOLDER_NAMES.keys()}
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = {key: [] for key in FOLDER_NAMES.keys()}

    # 단계 선택
    step = st.sidebar.selectbox(
        "단계를 선택하세요",
        ["1단계: 데이터 수집", "1-1단계: 데이터 수집 결과 요약", "2-3단계: 데이터 처리 및 탐색", "4단계: 데이터 분석 및 표현"]
    )

    if step == "1단계: 데이터 수집":
        step_1(BASE_FOLDER, FOLDER_NAMES)

    elif step == "1-1단계: 데이터 수집 결과 요약":
        step_2(FOLDER_NAMES)

    elif step == "2-3단계: 데이터 처리 및 탐색":
        step_3(BASE_FOLDER, FOLDER_NAMES)

    elif step == "4단계: 데이터 분석 및 표현":
        step_4(FOLDER_NAMES)


def get_image_urls_from_github(folder_name):
    """
    GitHub에서 폴더 내 이미지 파일을 자동으로 가져오는 함수
    """
    repo_url = "https://api.github.com/repos/GodTANKS/Galaxy/contents"
    folder_url = f"{repo_url}/{folder_name}"

    response = requests.get(folder_url)

    if response.status_code == 200:
        files = response.json()
        image_urls = []

        for file in files:
            if file['name'].lower().endswith(('.jpg', '.png', '.jpeg')):
                image_url = file['download_url']  # raw content URL
                image_urls.append(image_url)

        return image_urls
    else:
        st.error(f"GitHub API 오류: {response.status_code}")
        return []


def step_1(base_folder, folder_names):
    st.header("1단계: 데이터 수집")
    st.sidebar.write("### 은하 유형 선택")
    for folder_key, folder_label in folder_names.items():
        if st.sidebar.button(folder_label):
            st.session_state.selected_folder = folder_key

    if st.session_state.selected_folder is not None:
        selected_folder = st.session_state.selected_folder
        folder_label = folder_names[selected_folder]
        st.subheader(f"**{folder_label}** 데이터 수집")

        # GitHub에서 폴더 내 이미지 목록 가져오기
        images = get_image_urls_from_github(str(selected_folder))

        if images:
            st.write("**은하 이미지 번호를 입력하세요 (예: 이미지1-이미지10, 이미지15)**")
            image_range = st.text_area("이미지 번호 입력", "")
            if st.button(f"{folder_label} 이미지 수집"):
                if image_range:
                    collect_images(images, selected_folder, image_range)

            # 이미지 미리보기
            st.write("**해당 폴더의 이미지 미리보기**")
            cols = st.columns(3)
            for i, img_url in enumerate(images):
                image_name = img_url.split('/')[-1]
                with cols[i % 3]:
                    st.image(img_url, caption=f"이미지 {i + 1}: {image_name}", use_container_width=True)
        else:
            st.warning(f"{folder_label} 폴더에 이미지가 없습니다.")


def collect_images(images, folder_key, image_range):
    try:
        selected_images = []
        for part in image_range.split(","):
            part = part.strip()
            if "-" in part:
                start, end = map(int, part.replace("이미지", "").split("-"))
                selected_images.extend(range(start, end + 1))
            else:
                selected_images.append(int(part.replace("이미지", "")))

        for index in selected_images:
            if index <= len(images):
                selected_image_name = os.path.basename(images[index - 1])
                if selected_image_name not in st.session_state.collected_data[folder_key]:
                    st.session_state.collected_data[folder_key].append(selected_image_name)
                    st.success(f"{selected_image_name}이(가) 수집되었습니다.")
            else:
                st.warning(f"이미지 번호 {index}는 존재하지 않습니다.")
    except ValueError:
        st.error("입력 형식이 잘못되었습니다. 예: 이미지1-이미지10, 이미지15")


def step_2(folder_names):
    st.header("1-1단계: 데이터 수집 결과 요약")
    for folder, label in folder_names.items():
        st.write(f"- **{label} ({folder})**: {len(st.session_state.collected_data[folder])}개 이미지")


def step_3(base_folder, folder_names):
    st.header("2-3단계: 데이터 처리 및 탐색")
    if any(len(images) > 0 for images in st.session_state.collected_data.values()):
        selected_folder = st.selectbox(
            "데이터 처리할 은하 유형을 선택하세요",
            options=[key for key, images in st.session_state.collected_data.items() if images],
            format_func=lambda x: folder_names[x],
        )

        if selected_folder:
            resize_option = st.checkbox("이미지 크기 표준화 (50x50) (미선택시 데이터 분석 및 표현에서 오류발생)")
            is_grayscale = st.checkbox("흑백 이미지로 처리 (미선택시 칼라 이미지로 처리)", value=False)
            transformations = st.multiselect(
                "추가할 이미지 변환 선택",
                [
                    "45도 회전",
                    "90도 회전",
                    "135도 회전",
                    "180도 회전",
                    "225도 회전",
                    "270도 회전",
                    "315도 회전",
                    "좌우 대칭",
                    "좌우 대칭 및 45도 회전",
                    "좌우 대칭 및 90도 회전",
                    "좌우 대칭 및 135도 회전",
                    "좌우 대칭 및 180도 회전",
                    "좌우 대칭 및 225도 회전",
                    "좌우 대칭 및 270도 회전",
                    "좌우 대칭 및 315도 회전",
                ],
                default=[]
            )

            if st.button("데이터 처리 시작"):
                # 1단계에서 이미 로컬에 수집된 이미지 리스트를 가져옵니다
                image_names = st.session_state.collected_data[selected_folder]
                processed_images = process_images(
                    selected_folder, base_folder,
                    image_names, resize_option, transformations, not is_grayscale
                )
                st.session_state.processed_data[selected_folder] = processed_images
                st.success("데이터 처리가 완료되었습니다.")

                # 처리된 이미지들 표시 (3개씩 가로 나열)
                st.write("**데이터 탐색: 처리된 이미지들**")
                cols = st.columns(3)  # 가로 3개로 나누기
                for i, img_data in enumerate(processed_images):
                    with cols[i % 3]:
                        st.image(img_data, use_container_width=True)
    else:
        st.warning("수집된 데이터가 없습니다. 먼저 1단계에서 데이터 수집을 하세요.")


def process_images(folder_key, base_folder, image_names, resize, transformations, is_color=True):
    processed_images = []

    for image_name in image_names:
        # GitHub URL이 blob 형식이라면 raw URL로 변경
        image_url = f"https://github.com/GodTANKS/Galaxy/raw/main/{folder_key}/{image_name}"

        # raw URL로 변경된 이미지 URL 다운로드
        img = download_image(image_url)

        if img:
            if resize:
                img = img.resize((50, 50))

            # 이미지가 칼라인지 흑백인지 확인하여 처리
            if is_color:
                img = img.convert("RGB")  # 칼라 이미지로 변환
            else:
                img = img.convert("L")  # 흑백 이미지로 변환

            # 원본 이미지 추가
            processed_images.append(np.array(img))

            # 선택된 변환 적용
            for transformation in transformations:
                if transformation == "45도 회전":
                    rotated_img = img.rotate(45)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "90도 회전":
                    rotated_img = img.rotate(90)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "135도 회전":
                    rotated_img = img.rotate(135)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "180도 회전":
                    rotated_img = img.rotate(180)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "225도 회전":
                    rotated_img = img.rotate(225)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "270도 회전":
                    rotated_img = img.rotate(270)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "315도 회전":
                    rotated_img = img.rotate(315)
                    processed_images.append(np.array(rotated_img))
                elif transformation == "좌우 대칭":
                    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    processed_images.append(np.array(flipped_img))

    # 처리된 이미지와 개수 출력
    st.write(f"**데이터 탐색: 처리된 이미지 개수: {len(processed_images)}**")
    return processed_images


def download_image(url):
    """
    GitHub raw URL에서 이미지를 다운로드하여 반환하는 함수
    """
    try:
        # 이미지를 다운로드하고, 이미지 데이터로 변환
        response = requests.get(url)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))  # 이미지를 열기
            img.verify()  # 이미지가 유효한지 검증 (이 과정에서 오류가 발생하면 이미지가 아님)
            return Image.open(BytesIO(response.content))  # 이미지가 유효하면 반환
        else:
            st.error(f"이미지 다운로드 실패. 상태 코드: {response.status_code}, URL: {url}")
            return None
    except Exception as e:
        st.error(f"이미지 다운로드 중 오류 발생: {e}")
        return None


def visualize_misclassified_images(X, y_true, y_pred, title, folder_names, num_images=30):
    misclassified_indices = np.where(y_true != y_pred)[0]
    if len(misclassified_indices) == 0:
        st.write(f"{title}: 모든 예측이 정확합니다!")
        return

    st.write(f"{title}: 잘못 분류된 이미지 예시")

    num_images = min(num_images, len(misclassified_indices))

    cols = 5  # 🔹 가로(열) 개수를 5으로 고정
    rows = -(-num_images // cols)  # 🔹 올림 연산으로 행(row) 개수 자동 계산

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))  # 🔹 전체 크기 조절

    for i, idx in enumerate(np.random.choice(misclassified_indices, num_images, replace=False)):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]  # 🔹 행, 열에 맞게 배치
        ax.imshow(X[idx].squeeze(), cmap='gray' if X.shape[-1] == 1 else None)

        # ✅ folder_names이 리스트인지 딕셔너리인지 확인하고 올바르게 접근
        try:
            if isinstance(folder_names, dict):
                true_label = folder_names[str(y_true[idx])]  # 딕셔너리는 문자열 키로 변환
                pred_label = folder_names[str(y_pred[idx])]
            else:  # 리스트인 경우
                true_label = folder_names[y_true[idx]]
                pred_label = folder_names[y_pred[idx]]

            ax.set_title(f"True: {true_label}\nPredicted: {pred_label}", fontsize=10)
        except KeyError:
            ax.set_title("잘못된 레이블 값", fontsize=10)

        ax.axis('off')

    # 🔹 빈 칸 처리 (만약 num_images가 6의 배수가 아니면 빈 칸이 생길 수 있음)
    for j in range(num_images, rows * cols):
        fig.delaxes(axes.flatten()[j])  # 빈 공간 제거

    st.pyplot(fig)

def step_4(folder_names):
    st.header("4단계: 데이터 분석 및 표현")

    if any(len(images) > 0 for images in st.session_state.processed_data.values()):
        # 데이터 분할
        st.subheader("1. 데이터 분할")
        train_percentage = st.slider("훈련 데이터 비율 (%)", 60, 100, 80)
        val_percentage = st.slider("검증 데이터 비율 (%)", 0, 40, 10)
        test_percentage = 100 - (train_percentage + val_percentage)

        st.write(f"훈련 데이터 비율: {train_percentage}%")
        st.write(f"검증 데이터 비율: {val_percentage}%")
        st.write(f"테스트 데이터 비율: {test_percentage}%")

        # 이미지 및 레이블 준비
        images, labels = [], []
        for label, image_list in st.session_state.processed_data.items():
            label = str(label)
            images.extend(image_list)
            labels.extend([int(label)] * len(image_list))

        # 이미지와 레이블 수 확인
        if len(images) != len(labels):
            st.error("이미지와 레이블 수가 일치하지 않습니다.")
            return

        # 🔹 사용자가 선택한 이미지 모드에 맞게 데이터 변환
        first_image = images[0]
        img_shape = first_image.shape  # 이미지의 원본 크기 확인

        if len(img_shape) == 3:  # 칼라 이미지 (H, W, 3)
            is_color = True
            X = np.array(images, dtype=np.float32) / 255.0  # RGB 정규화
        else:  # 흑백 이미지 (H, W)
            is_color = False
            X = np.array(images, dtype=np.float32).reshape(-1, 50, 50, 1) / 255.0  # Grayscale 정규화

        y = tf.keras.utils.to_categorical(np.array(labels), num_classes=5)

        # 데이터 분할
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1 - train_percentage / 100, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_percentage / (test_percentage + val_percentage),
            random_state=42
        )

        st.write(f"훈련 데이터: {len(X_train)}개, 검증 데이터: {len(X_val)}개, 테스트 데이터: {len(X_test)}개")

        # 모델 구축
        st.subheader("2. 모델 구성")
        model = build_cnn_model(is_color)

        st.write("**모델 구조**:")  # 모델 구조 표시
        summary_string = []
        model.summary(print_fn=lambda x: summary_string.append(x))
        st.text("\n".join(summary_string))

        # 배치 크기와 에포크 설정 및 학습률
        batch_size = st.sidebar.slider("배치 크기 (Batch Size)", 8, 128, 32)
        epochs = st.sidebar.slider("에포크 수 (Epochs)", 1, 100, 5)
        learning_rate = st.sidebar.selectbox("학습률 (Learning Rate)", [0.001, 0.0001, 0.00001, 0.000001])
        st.write(f"선택된 학습률: {learning_rate}")

        # 버튼으로 모델 훈련 시작
        if st.button("모델 훈련 시작"):
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.004, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            # 실시간으로 훈련 과정을 보여주기 위한 콜백
            class StreamlitCallback(tf.keras.callbacks.Callback):
                def __init__(self):
                    super().__init__()
                    self.progress_text = st.empty()  # 빈 공간을 만들어 업데이트 방식 적용
                    self.logs_data = []  # 에포크별 로그를 저장할 리스트

                def on_epoch_end(self, epoch, logs=None):
                    logs = logs or {}
                    formatted_logs = f"Epoch {epoch + 1}/{epochs} | " + " | ".join(
                        [f"{key}: {value:.4f}" for key, value in logs.items()])
                    self.logs_data.append(formatted_logs)  # 새로운 로그를 리스트에 추가
                    self.progress_text.text("\n".join(self.logs_data))  # 전체 로그를 한 번에 갱신하여 가로로 쌓기

            st.subheader("3. 모델 훈련")
            start_time = time.time()
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[StreamlitCallback()],
                verbose=0
            )
            end_time = time.time()

            training_time = end_time - start_time
            st.write(f"훈련 시간: {training_time:.2f}초")

            # 훈련 결과 시각화
            plot_training_results(history)

            # 검증 데이터 혼동 행렬
            st.subheader("4. 모델 검증")
            y_val_pred = model.predict(X_val).argmax(axis=1)
            y_val_true = y_val.argmax(axis=1)
            st.write("### 검증 데이터에 대한 혼동 행렬")
            plot_confusion_matrix(y_val_true, y_val_pred, folder_names)

            # 테스트 데이터 혼동 행렬
            st.subheader("5. 모델 테스트")
            y_test_pred = model.predict(X_test).argmax(axis=1)
            y_test_true = y_test.argmax(axis=1)
            st.write("### 테스트 데이터에 대한 혼동 행렬")
            plot_confusion_matrix(y_test_true, y_test_pred, folder_names)

            # 테스트 결과 요약 그래프
            st.subheader("6. 테스트 결과 요약")
            test_results = model.evaluate(X_test, y_test, verbose=0)
            metrics = ["Loss", "Accuracy"]
            values = test_results[:2]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(metrics, values, color=["#1f77b4", "#ff7f0e"])
            ax.set_ylim([0, 1.0])
            ax.set_ylabel("Value")
            ax.set_title("Test Results Summary")
            for i, v in enumerate(values):
                ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
            st.pyplot(fig)

            # 잘못 분류된 이미지 시각화 추가
            st.subheader("7. 잘못 분류된 이미지 시각화")
            visualize_misclassified_images(X_train, y_train.argmax(axis=1), model.predict(X_train).argmax(axis=1), "훈련 데이터",
                                           folder_names)
            visualize_misclassified_images(X_val, y_val.argmax(axis=1), y_val_pred, "검증 데이터", folder_names)
            visualize_misclassified_images(X_test, y_test.argmax(axis=1), y_test_pred, "테스트 데이터", folder_names)

    else:
        st.warning("처리된 데이터가 없습니다. 먼저 3단계에서 데이터 처리를 하세요.")


def build_cnn_model(is_color):
    st.sidebar.header("모델 구성")

    # CNN 레이어 수, 필터 크기, 활성화 함수 선택
    conv_layers = st.sidebar.slider("전반부 콘볼루션 레이어 개수 (Convolutional Layers)", 1, 6, 2)
    filter_options = [8, 16, 32, 64, 128, 256, 512, 1024]  # 선택 가능한 필터 값
    filters = [
        st.sidebar.selectbox(
            f"전반부 필터(채널) 개수 (Filters, 예: 32는 이미지 채널 32개 의미) (레이어 {i + 1})",
            filter_options,
            index=filter_options.index(32)  # 기본값: 32
        ) for i in range(conv_layers)
    ]
    pool_options = [2, 3, 4, 5, 6]  # 선택 가능한 풀링 값
    pool_size = [
        st.sidebar.selectbox(
            f"전반부 최대 풀링 크기 (Max Pooling Size, 예: 2는 2x2 픽셀 의미) (레이어 {i + 1})",
            pool_options,
            index=pool_options.index(2)  # 기본값: 2
        ) for i in range(conv_layers)
    ]

    activation_functions = [
        st.sidebar.selectbox(f"전반부 활성화 함수 (Activation Function) (레이어 {i + 1})", ["relu", "sigmoid", "tanh", "softmax"]) for
        i in range(conv_layers)
    ]

    # 마지막을 제외한 후반부 완전 연결(dense) 레이어 개수와 Dropout 레이어 개수 선택
    dense_layers = st.sidebar.slider("마지막을 제외한 후반부 완전 연결(Dense) 레이어 개수", 1, 3, 2)
    dropout_layers = st.sidebar.slider("드롭아웃(Dropout) 레이어 개수", 0, 3, 2)

    # 드롭아웃(Dropout) 비율 선택 (0과 1 사이)
    dropout_rate = st.sidebar.slider("드롭아웃(Dropout) 비율", 0.0, 1.0, 0.5, step=0.05)

    # 각 후반부 완전 연결(Dense) 레이어의 크기 선택
    dense_units = [
        st.sidebar.selectbox(f"후반부 완전 연결(Dense) 레이어 {i+1} 크기", [8, 16, 32, 64, 128, 256, 512, 1024], index=3)  # 기본값 64
        for i in range(dense_layers)
    ]

    # 각 후반부 완전 연결(Dense) 레이어의 활성화 함수 선택
    dense_activation_functions = [
        st.sidebar.selectbox(f"후반부 완전 연결(Dense) 레이어 {i+1} 활성화 함수", ["relu", "sigmoid", "tanh", "softmax"]) for i in range(dense_layers)
    ]

    # 마지막 완전 연결(Dense) 레이어의 활성화 함수 선택 옵션을 완전 연결(Dense) 레이어 활성화 함수 뒤로 이동
    last_activation = st.sidebar.selectbox("마지막 완전 연결(Dense) 레이어의 활성화 함수 (Activation Function for Last Dense)", ["softmax", "relu", "sigmoid", "tanh"])

    model = tf.keras.Sequential()
    input_channels = 3 if is_color else 1  # 칼라 이미지일 경우 3, 흑백일 경우 1

    model.add(tf.keras.layers.InputLayer(input_shape=(50, 50, input_channels)))  # 입력 이미지 크기

    # 각 레이어 구성
    for i in range(conv_layers):
        model.add(tf.keras.layers.Conv2D(filters[i], (3, 3), activation=None, padding='same'))
        model.add(tf.keras.layers.BatchNormalization())  # 배치 정규화 추가
        model.add(tf.keras.layers.Activation(activation_functions[i]))  # 활성화 함수 추가
        model.add(tf.keras.layers.MaxPooling2D(pool_size[i], padding='same'))

    model.add(tf.keras.layers.GlobalAveragePooling2D())  # GlobalAveragePooling2D 추가

    # 마지막을 제외한 후반부 완전 연결(Dense) 레이어 및 드롭 아웃(Dropout) 레이어 구성
    for i in range(dense_layers):
        model.add(tf.keras.layers.Dense(dense_units[i], activation=None))  # 활성화 함수는 나중에 추가
        model.add(tf.keras.layers.BatchNormalization())  # 배치 정규화 추가
        model.add(tf.keras.layers.Activation(dense_activation_functions[i]))  # 활성화 함수 추가
        if i < dropout_layers:  # 드롭 아웃(Dropout) 레이어 추가
            model.add(tf.keras.layers.Dropout(dropout_rate))  # 선택한 드롭 아웃(Dropout) 비율 사용

    model.add(tf.keras.layers.Dense(5, activation=last_activation))  # 마지막 완전 연결(Dense) 레이어

    return model

def get_callbacks():
    # EarlyStopping 및 ModelCheckpoint 추가
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "best_model.h5", monitor="val_loss", save_best_only=True
    )
    return [early_stopping, model_checkpoint]


def plot_confusion_matrix(y_true, y_pred, folder_names):
    cm = confusion_matrix(y_true, y_pred)  # 혼동 행렬 생성
    fig, ax = plt.subplots(figsize=(8, 6))  # 플롯 크기 설정
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=folder_names.values(),
                yticklabels=folder_names.values())
    ax.set_xlabel('Predicted')  # x축 레이블
    ax.set_ylabel('True')       # y축 레이블
    ax.set_title('Confusion Matrix')  # 플롯 제목
    st.pyplot(fig)  # Streamlit에 플롯 표시

def plot_training_results(history):
    st.subheader("훈련 결과 요약")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 정확도 그래프
    ax[0].plot(history.history['accuracy'], label='Training Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()

    # 손실 그래프
    ax[1].plot(history.history['loss'], label='Training Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    st.pyplot(fig)

if __name__ == "__main__":
    main()

