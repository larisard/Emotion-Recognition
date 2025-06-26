import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2 #type:ignore
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input #type:ignore
from tensorflow.keras.models import Sequential #type:ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout #type:ignore
from tensorflow.keras import callbacks #type:ignore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==================== Configuração ====================
def configurar_ambiente():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ==================== Carregar Vídeos com Haar Cascade ====================

haar_cascade_path = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def carregar_frames_videos(caminho_base, num_frames_por_video=20, tamanho_imagem=(48, 48)):
    todos_frames = []
    todos_rotulos = []
    nomes_classes = []
    mapa_classes = {}
    indice_classe_atual = 0

    if not os.path.isdir(caminho_base):
        return np.array([]), np.array([]), []

    subdirs = sorted([d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))])

    for subdir_nome in tqdm(subdirs, desc="Processando Subdiretórios"):
        try:
            nome_classe = subdir_nome.split("_")[1].split("-")[1]
        except (IndexError, ValueError):
            continue

        if nome_classe not in mapa_classes:
            mapa_classes[nome_classe] = indice_classe_atual
            nomes_classes.append(nome_classe)
            indice_classe_atual += 1
        rotulo_classe = mapa_classes[nome_classe]

        caminho_subdir = os.path.join(caminho_base, subdir_nome)
        arquivos_video = sorted([f for f in os.listdir(caminho_subdir) if os.path.isfile(os.path.join(caminho_subdir, f))])

        for nome_arquivo in arquivos_video:
            caminho_video = os.path.join(caminho_subdir, nome_arquivo)
            cap = cv2.VideoCapture(caminho_video)

            if not cap.isOpened():
                continue

            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames_video <= 0:
                cap.release()
                continue

            if total_frames_video < num_frames_por_video:
                indices_frames = np.arange(total_frames_video)
            else:
                indices_frames = np.linspace(0, total_frames_video - 1, num_frames_por_video, dtype=int)

            frame_count = 0
            capturados_temp = {}
            while cap.isOpened() and frame_count < total_frames_video:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count in indices_frames:
                    capturados_temp[frame_count] = frame
                frame_count += 1
            cap.release()

            frames_selecionados_para_video = []
            for idx_target in indices_frames:
                if idx_target in capturados_temp:
                    frame_to_use = capturados_temp[idx_target]
                    frame_cinza = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(frame_cinza, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face_roi = frame_cinza[y:y+h, x:x+w]
                        if face_roi.size == 0:
                            frame_processado = cv2.resize(frame_cinza, tamanho_imagem, interpolation=cv2.INTER_AREA)
                        else:
                            frame_processado = cv2.resize(face_roi, tamanho_imagem, interpolation=cv2.INTER_AREA)
                    else:
                        frame_processado = cv2.resize(frame_cinza, tamanho_imagem, interpolation=cv2.INTER_AREA)
                    frames_selecionados_para_video.append(frame_processado)
                else:
                    if frames_selecionados_para_video:
                        frames_selecionados_para_video.append(frames_selecionados_para_video[-1])
                    else:
                        frames_selecionados_para_video = []
                        break

            if 0 < len(frames_selecionados_para_video) < num_frames_por_video:
                ultimo_frame_valido = frames_selecionados_para_video[-1]
                frames_selecionados_para_video.extend([ultimo_frame_valido] * (num_frames_por_video - len(frames_selecionados_para_video)))
            elif len(frames_selecionados_para_video) == 0:
                continue

            if len(frames_selecionados_para_video) == num_frames_por_video:
                todos_frames.extend(frames_selecionados_para_video)
                todos_rotulos.extend([rotulo_classe] * num_frames_por_video)

    if not todos_frames:
        return np.array([]), np.array([]), []

    return np.array(todos_frames), np.array(todos_rotulos), nomes_classes
# ==================== Extrator HOG ====================
class HOGFeatureExtractor:
    @staticmethod
    def calcularGradiente(imagem):
        imagem = cv2.GaussianBlur(imagem.astype(np.float32), (3, 3), 0)
        imagem = imagem.astype(np.float32)
        gradienteX = cv2.Sobel(imagem, cv2.CV_32F, 1, 0, ksize=1)
        gradienteY = cv2.Sobel(imagem, cv2.CV_32F, 0, 1, ksize=1)
        magnitude = np.sqrt(gradienteX**2 + gradienteY**2)
        angulo = np.arctan2(gradienteY, gradienteX) * (180 / np.pi)
        angulo[angulo < 0] += 180
        return magnitude, angulo

    @staticmethod
    def calcularHistogramaCelula(imagens, tamanho_imagem=(48,48), tamanho_bloco=2, tamanho_celula=6, bins=12):
        hog_vetores = []
        print(f"Calculando HOG customizado para {len(imagens)} frames...")
        for img in tqdm(imagens, desc="Custom HOG Features"):
            if img.shape != tamanho_imagem:
                img = cv2.resize(img, tamanho_imagem, interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32)

            magnitude, angulo = HOGFeatureExtractor.calcularGradiente(img)

            altura, largura = img.shape
            celula_x = largura // tamanho_celula
            celula_y = altura // tamanho_celula

            histograma_orientacoes = np.zeros((celula_y, celula_x, bins))

            for i in range(celula_y):
                for j in range(celula_x):
                    mag_celula = magnitude[i*tamanho_celula:(i+1)*tamanho_celula, j*tamanho_celula:(j+1)*tamanho_celula]
                    ang_celula = angulo[i*tamanho_celula:(i+1)*tamanho_celula, j*tamanho_celula:(j+1)*tamanho_celula]

                    hist = np.zeros(bins)
                    for y_cell in range(tamanho_celula):
                        for x_cell in range(tamanho_celula):
                            bin_idx = int(ang_celula[y_cell, x_cell] / (180.0 / bins)) % bins if bins > 0 else 0
                            hist[bin_idx] += mag_celula[y_cell, x_cell]
                    histograma_orientacoes[i, j] = hist

            blocos_y = celula_y - tamanho_bloco + 1
            blocos_x = celula_x - tamanho_bloco + 1
            blocos_normalizados = []

            for y_block in range(blocos_y):
                for x_block in range(blocos_x):
                    bloco = histograma_orientacoes[y_block:y_block+tamanho_bloco, x_block:x_block+tamanho_bloco].flatten()
                    norm = np.linalg.norm(bloco) + 1e-6
                    bloco_normalizado = bloco / norm
                    bloco_normalizado = np.minimum(bloco_normalizado, 0.2) # Clipping L2-Hys
                    norm = np.linalg.norm(bloco_normalizado) + 1e-6
                    bloco_normalizado = bloco_normalizado / norm
                    blocos_normalizados.append(bloco_normalizado)

            vetor_hog = np.concatenate(blocos_normalizados) if blocos_normalizados else np.array([])
            hog_vetores.append(vetor_hog)

        if hog_vetores:
            tamanho_ref = len(hog_vetores[0]) if hog_vetores else 0
            hog_vetores_consistentes = []
            for i, v in enumerate(hog_vetores):
                if len(v) == tamanho_ref:
                    hog_vetores_consistentes.append(v)
                else:
                    print(f"\nAviso HOG: Vetor {i} tem tamanho {len(v)}, esperado {tamanho_ref}. Preenchendo/truncando.")
                    vetor_ajustado = np.zeros(tamanho_ref)
                    len_min = min(len(v), tamanho_ref)
                    vetor_ajustado[:len_min] = v[:len_min]
                    hog_vetores_consistentes.append(vetor_ajustado)
            hog_vetores_np = np.array(hog_vetores_consistentes)
        else:
            hog_vetores_np = np.array([])

        if hog_vetores_np.size > 0:
             print(f"Shape features HOG (custom): {hog_vetores_np.shape}")
        else:
             print("Nenhum vetor HOG foi gerado.")
        return hog_vetores_np

# ==================== Extrator CNN (MobileNetV2) ====================
mobilenet = None
def inicializar_mobilenet(tamanho_imagem):
    global mobilenet
    if mobilenet is None:
        input_shape_tf = (tamanho_imagem[1], tamanho_imagem[0], 3)
        mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                                input_shape=input_shape_tf)
        mobilenet.trainable = False
        print(f"MobileNetV2 inicializado com input_shape={input_shape_tf}")

def extrair_cnn(imagens_gray):
    global mobilenet

    if imagens_gray.ndim == 2: 
        imagens_gray = np.expand_dims(imagens_gray, axis=0) 
    if imagens_gray.ndim == 3:
        imagens_gray = np.expand_dims(imagens_gray, axis=-1) 

    if imagens_gray.shape[-1] == 1:
        imgs_rgb = np.repeat(imagens_gray, 3, axis=-1)
    elif imagens_gray.shape[-1] == 3:
        imgs_rgb = imagens_gray
    else:
        raise ValueError(f"Formato de imagem inesperado para CNN: {imagens_gray.shape}")

    imgs_rgb = imgs_rgb.astype(np.float32)
    imgs_rgb = preprocess_input(imgs_rgb)

    print(f"Extraindo features CNN para {len(imgs_rgb)} frames...")
    extracao = mobilenet.predict(imgs_rgb, batch_size=32, verbose=0)
    print(f"Shape features CNN: {extracao.shape}")
    return extracao

# ==================== Preparar Dados para LSTM ====================
def preparar_dados_video_LSTM(frames_cinza, rotulos, seq_length, tamanho_imagem=(48, 48)):

    num_total_frames = frames_cinza.shape[0]

    if num_total_frames % seq_length != 0:
        print(f"\nAviso LSTM: Número total de frames ({num_total_frames}) não é divisível por tamanho_sequencia ({seq_length}).")
        num_videos = num_total_frames // seq_length
        if num_videos == 0:
             print(f"\nErro LSTM: Menos frames ({num_total_frames}) que tamanho_sequencia ({seq_length}). Impossível criar sequências.")
             return None, None, None
        frames_cinza = frames_cinza[:num_videos * seq_length]
        rotulos = rotulos[:num_videos * seq_length]
        print(f"Usando {num_videos * seq_length} frames para formar {num_videos} sequências.")
    else:
        num_videos = num_total_frames // seq_length

    print(f"Preparando {num_videos} sequências de {seq_length} frames cada para LSTM.")

    # --- Extração de Features ---
    features_hog = HOGFeatureExtractor.calcularHistogramaCelula(frames_cinza, tamanho_imagem=tamanho_imagem)
    if features_hog is None or features_hog.size == 0:
        print("\nErro LSTM: Falha ao extrair features HOG.")
        return None, None, None

    if mobilenet is None:
        print("\nInicializando MobileNetV2 antes da extração CNN...")
        inicializar_mobilenet(tamanho_imagem)

    try:
        features_cnn = extrair_cnn(frames_cinza)
        if features_cnn is None or features_cnn.size == 0:
             print("\nErro LSTM: Falha ao extrair features CNN.")
             return None, None, None
    except Exception as e:
        print(f"\nErro LSTM durante a extração de features CNN: {e}")
        return None, None, None

    if features_hog.shape[0] != frames_cinza.shape[0] or features_cnn.shape[0] != frames_cinza.shape[0]:
        print(f"\nErro LSTM: Inconsistência no número de features extraídas. "
              f"Frames: {frames_cinza.shape[0]}, HOG: {features_hog.shape[0]}, CNN: {features_cnn.shape[0]}")
        # Tenta usar o mínimo de features consistentes
        min_len = min(features_hog.shape[0], features_cnn.shape[0], frames_cinza.shape[0])
        if min_len > 0 and min_len % seq_length == 0:
            print(f"Ajustando para usar {min_len} frames/features consistentes.")
            features_hog = features_hog[:min_len]
            features_cnn = features_cnn[:min_len]
            frames_cinza = frames_cinza[:min_len]
            rotulos = rotulos[:min_len]
            num_videos = min_len // seq_length
        else:
            print("Não foi possível ajustar. Abortando preparação LSTM.")
            return None, None, None

    # --- Combinação de Features ---
    print(f"Concatenando HOG ({features_hog.shape}) e CNN ({features_cnn.shape}) features...")
    try:
        features_combinadas = np.concatenate([features_hog, features_cnn], axis=1)
    except ValueError as e:
        print(f"\nErro LSTM ao concatenar features: {e}. Shapes: HOG={features_hog.shape}, CNN={features_cnn.shape}")
        return None, None, None
    print(f"Shape features combinadas (antes de reshape): {features_combinadas.shape}")

    # --- Remodelar para Sequências LSTM ---
    num_features_combinadas = features_combinadas.shape[1]
    try:
        X_seq = features_combinadas.reshape((num_videos, seq_length, num_features_combinadas))
    except ValueError as e:
        print(f"\nErro LSTM ao remodelar features combinadas para sequências: {e}")
        print(f"Tentando remodelar {features_combinadas.shape} para ({num_videos}, {seq_length}, {num_features_combinadas})")
        return None, None, None

    # --- Obter Rótulos para Sequências ---
    y_seq = rotulos[::seq_length]
    if len(y_seq) != num_videos:
        print(f"\nErro LSTM: Inconsistência no número de rótulos de sequência. "
              f"Esperado: {num_videos}, Obtido: {len(y_seq)}")
        y_seq = rotulos[:num_videos*seq_length:seq_length]
        if len(y_seq) != num_videos:
             print("Ajuste de rótulos falhou. Abortando.")
             return None, None, None

    print(f"Shape final X_seq (LSTM input): {X_seq.shape}")
    print(f"Shape final y_seq (LSTM labels): {y_seq.shape}")

    return X_seq, y_seq, features_combinadas

# ==================== Construir Modelo LSTM ====================
def construir_modelo_LSTM(input_shape, num_classes):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Modelo LSTM construído:")
    model.summary()
    return model

# ==================== Treinamento e Avaliação ====================
def treinar_avaliar_modelo(X_treino, y_treino, X_val, y_val, X_teste, y_teste,
                           num_classes, classes_emocao, X_teste_frames_orig=None, seq_length=None):


    input_shape_lstm = (X_treino.shape[1], X_treino.shape[2])
    modelo = construir_modelo_LSTM(input_shape_lstm, num_classes)
    
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1),
    ]
    
    print("\nIniciando treinamento do modelo LSTM...")
    historico = modelo.fit(X_treino, y_treino,
                           epochs=100,
                           batch_size=16,
                           validation_data=(X_val, y_val),
                           callbacks=callbacks_list,
                           verbose=1)

    print("\nTreinamento concluído. Avaliando no conjunto de teste...")
    loss, accuracy = modelo.evaluate(X_teste, y_teste, verbose=0)
    print(f"Acurácia: {accuracy:.4f}")

    y_pred_prob = modelo.predict(X_teste, batch_size=16, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)

    precisao = precision_score(y_teste, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_teste, y_pred, average='weighted', zero_division=0)
    matriz_conf = confusion_matrix(y_teste, y_pred)

    print(f"Precisão: {precisao:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Matriz de Confusão:")
    print(matriz_conf)

    visualizar_resultados(historico, matriz_conf, classes_emocao)

    if X_teste_frames_orig is not None and seq_length is not None:
        print("\nGerando visualização de amostra de predições...")
        visualizar_amostra_video(X_teste_frames_orig, y_teste, y_pred, classes_emocao, seq_length)
    else:
        print("\nFrames originais ou seq_length não fornecidos, pulando visualização de amostra.")
        
    
    return modelo, historico

# ==================== Funções de Visualização e Avaliação ====================
def visualizar_resultados(historico, matriz_confusao, classes_emocao):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Acurácia Treino')
    plt.plot(historico.history['val_accuracy'], label='Acurácia Validação')
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Perda Treino')
    plt.plot(historico.history['val_loss'], label='Perda Validação')
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Gráfico do histórico de treinamento salvo em 'training_history.png'")
    plt.close()

    plt.figure(figsize=(10, 8))
    sum_axis1 = matriz_confusao.sum(axis=1)[:, np.newaxis]
    matriz_confusao_norm = np.divide(matriz_confusao.astype('float'), sum_axis1,
                                     out=np.zeros_like(matriz_confusao, dtype=float),
                                     where=sum_axis1!=0)

    sns.heatmap(matriz_confusao_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes_emocao, yticklabels=classes_emocao)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusão Normalizada')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Matriz de confusão salva em 'confusion_matrix.png'")
    plt.close()

def visualizar_amostra_video(X_seq_original_frames, true_labels, predicted_labels, classes_emocao, seq_length):

    nome_arquivo = 'amostra_predicoes_video.png'

    if X_seq_original_frames is None or X_seq_original_frames.size == 0:
        print("Aviso: Frames originais não disponíveis para visualização de amostra.")
        return

    if X_seq_original_frames.ndim != 4 or X_seq_original_frames.shape[1] != seq_length:
         print(f"Aviso: Shape inesperado para X_seq_original_frames: {X_seq_original_frames.shape}. "
               f"Esperado (~, {seq_length}, H, W). Pulando visualização.")
         num_videos_esperado = len(true_labels)
         if X_seq_original_frames.ndim == 3 and X_seq_original_frames.shape[0] == num_videos_esperado * seq_length:
             try:
                 altura, largura = X_seq_original_frames.shape[1], X_seq_original_frames.shape[2]
                 X_seq_original_frames = X_seq_original_frames.reshape((num_videos_esperado, seq_length, altura, largura))
                 print("Remodelagem de X_seq_original_frames bem-sucedida.")
             except ValueError as e:
                 print(f"Erro ao tentar remodelar X_seq_original_frames: {e}. Pulando visualização.")
                 return
         else:
              return

    plt.figure(figsize=(15, 5))
    num_amostras_total = X_seq_original_frames.shape[0]
    num_amostras_para_mostrar = min(5, num_amostras_total)

    if num_amostras_para_mostrar == 0:
        print("Nenhuma amostra disponível para visualização.")
        plt.close()
        return

    indices_amostra = np.random.choice(num_amostras_total, num_amostras_para_mostrar, replace=False)

    for i, idx in enumerate(indices_amostra):
        ultimo_frame = X_seq_original_frames[idx, -1, :, :]
        plt.subplot(1, num_amostras_para_mostrar, i + 1)
        plt.imshow(ultimo_frame, cmap='gray')

        try:
            true_label_idx = int(true_labels[idx])
            pred_label_idx = int(predicted_labels[idx])
            if 0 <= true_label_idx < len(classes_emocao) and 0 <= pred_label_idx < len(classes_emocao):
                titulo = f"Verdadeiro: {classes_emocao[true_label_idx]}\nPredito: {classes_emocao[pred_label_idx]}"
            else:
                titulo = f"Erro: Índice de Rótulo Inválido\nTrue: {true_label_idx}, Pred: {pred_label_idx}"
        except (IndexError, ValueError, TypeError) as e:
            titulo = f"Erro Rótulo ({type(e).__name__})\nTrue: {true_labels[idx]}, Pred: {predicted_labels[idx]}"

        plt.title(titulo)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(nome_arquivo)
    print(f"Visualização de amostra salva em '{nome_arquivo}'")
    plt.close()

# ==================== Função Principal ====================
def main():
    configurar_ambiente()

    # --- Parâmetros ---
    CAMINHO_BASE_DADOS = 'S4 Stimuli (Video-clips)'
    NUM_FRAMES_POR_VIDEO = 20
    TAMANHO_IMAGEM = (48, 48)
    SEQ_LENGTH = NUM_FRAMES_POR_VIDEO
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2 

    # --- Carregar e Processar Dados ---
    print("Fase 1: Carregando vídeos e extraindo frames/faces com Haar Cascade...")
    frames_processados, rotulos, nomes_classes = carregar_frames_videos(
        CAMINHO_BASE_DADOS,
        num_frames_por_video=NUM_FRAMES_POR_VIDEO,
        tamanho_imagem=TAMANHO_IMAGEM
    )

    if frames_processados.size == 0:
        print("Nenhum frame foi carregado. Verifique o diretório e os vídeos. Encerrando.")
        return

    num_classes = len(nomes_classes)
    print(f"Número de classes detectadas: {num_classes} ({', '.join(nomes_classes)}) ")
    if num_classes < 2:
        print("Erro: Menos de duas classes encontradas. O treinamento requer pelo menos duas classes.")
        return

    # --- Preparar Dados para LSTM ---
    print("\nFase 2: Preparando dados para LSTM (Extração HOG/CNN e Sequenciamento)...")
    X_seq, y_seq, _ = preparar_dados_video_LSTM(
        frames_processados,
        rotulos,
        seq_length=SEQ_LENGTH,
        tamanho_imagem=TAMANHO_IMAGEM
    )


    # --- Divisão Treino/Validação/Teste ---
    print("\nFase 3: Dividindo os dados em conjuntos de Treino, Validação e Teste...")
    
    # Mapeamento de rótulos para garantir estratificação correta
    map_rotulos_unicos = {label: i for i, label in enumerate(np.unique(y_seq))}
    y_seq_mapped = np.array([map_rotulos_unicos[label] for label in y_seq])

    try:
        val_test_size = VALIDATION_SIZE + TEST_SIZE # Tamanho combinado para Val+Teste
        train_size = 1.0 - val_test_size

        # Divide em Treino e (Val + Teste)
        X_treino_seq, X_val_teste_seq, y_treino_seq, y_val_teste_seq = train_test_split(
            X_seq, y_seq_mapped, train_size=train_size, random_state=42, stratify=y_seq_mapped
        )

        # Calcula a proporção de teste relativa ao conjunto (Val + Teste)
        test_size_relative = TEST_SIZE / val_test_size

        # Divide (Val + Teste) em Validação e Teste
        X_val_seq, X_teste_seq, y_val_seq, y_teste_seq = train_test_split(
            X_val_teste_seq, y_val_teste_seq, test_size=test_size_relative, random_state=42, stratify=y_val_teste_seq
        )

        print(f"Tamanho Treino: {len(X_treino_seq)} sequências")
        print(f"Tamanho Validação: {len(X_val_seq)} sequências")
        print(f"Tamanho Teste: {len(X_teste_seq)} sequências")

        # --- Preparar Frames Originais para Visualização do Teste ---
        X_teste_frames_para_visualizar = None
        print("Aviso: Visualização de amostra de teste usará frames processados (não originais)." )
        # Tentar obter os frames processados correspondentes ao conjunto de teste
        try:
            # Encontra os índices originais das sequências de teste
            indices_teste_orig = []
            for i in range(len(X_seq)):
                 # Compara a sequência original com as sequências no conjunto de teste
                 if any(np.array_equal(X_teste_seq[j], X_seq[i]) for j in range(len(X_teste_seq))):
                     indices_teste_orig.append(i)

            if indices_teste_orig:
                indices_frames_teste = []
                for idx_seq in indices_teste_orig:
                    inicio_frame = idx_seq * SEQ_LENGTH
                    fim_frame = inicio_frame + SEQ_LENGTH
                    indices_frames_teste.extend(range(inicio_frame, fim_frame))

                if len(indices_frames_teste) == len(X_teste_seq) * SEQ_LENGTH:
                    frames_teste_proc = frames_processados[indices_frames_teste]
                    altura, largura = TAMANHO_IMAGEM
                    X_teste_frames_para_visualizar = frames_teste_proc.reshape((len(X_teste_seq), SEQ_LENGTH, altura, largura))
                    print("Frames processados para visualização do teste preparados.")
                else:
                    print("Inconsistência ao tentar extrair frames de teste para visualização (contagem).")
            else:
                 print("Não foi possível mapear sequências de teste aos frames originais (índices).")

        except Exception as e:
            print(f"Erro ao preparar frames para visualização: {e}")

    except ValueError as e:
        print(f"Erro durante a divisão dos dados: {e}")
        print("Isso pode ocorrer se houver poucas amostras por classe para estratificação.")
        return

    # --- Treinamento e Avaliação ---
    print("\nFase 4: Treinando e Avaliando o Modelo LSTM...")
    modelo_treinado, historico_treinamento = treinar_avaliar_modelo(
        X_treino_seq, y_treino_seq,
        X_val_seq, y_val_seq,
        X_teste_seq, y_teste_seq,
        num_classes,
        nomes_classes,
        X_teste_frames_para_visualizar,
        SEQ_LENGTH
    )

    if modelo_treinado:
        print("\nModelo treinado com sucesso!")
        modelo_treinado.save("classificador_emocaoLS.keras") 

    print("\nExecução concluída.")

# --- Ponto de Entrada ---
if __name__ == "__main__":
    main()

