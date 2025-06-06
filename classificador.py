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
    
# ==================== Carregar Vídeos ====================
def carregar_frames_videos(caminho_base, num_frames_por_video=20, tamanho_imagem=(48, 48)):

    todos_frames = []
    todos_rotulos = []
    nomes_classes = []
    mapa_classes = {}
    indice_classe_atual = 0


    subdirs = sorted([d for d in os.listdir(caminho_base) if os.path.isdir(os.path.join(caminho_base, d))])

    for subdir_nome in subdirs:
        try:
            partes_nome = subdir_nome.split("_")
            if len(partes_nome) < 2:
                 raise IndexError("Formato de nome de subdiretório inválido")
            partes_emocao = partes_nome[1].split("-")
            if len(partes_emocao) < 2:
                 raise IndexError("Formato de nome de emoção inválido")
            nome_classe = partes_emocao[1]
        except IndexError as e:
            print(f"Aviso: Ignorando subdiretório com nome inesperado ",
                  f"'{subdir_nome}': {e}")
            continue

        if nome_classe not in mapa_classes:
            mapa_classes[nome_classe] = indice_classe_atual
            nomes_classes.append(nome_classe)
            indice_classe_atual += 1
        rotulo_classe = mapa_classes[nome_classe]

        caminho_subdir = os.path.join(caminho_base, subdir_nome)
        arquivos_video = sorted([f for f in os.listdir(caminho_subdir) if os.path.isfile(os.path.join(caminho_subdir, f))])
        if not arquivos_video:
             continue

        for nome_arquivo in arquivos_video:
            caminho_video = os.path.join(caminho_subdir, nome_arquivo)
            cap = cv2.VideoCapture(caminho_video)


            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames_selecionados = []

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

            for idx_target in indices_frames:
                if idx_target in capturados_temp:
                    frame_to_use = capturados_temp[idx_target]
                    frame_cinza = cv2.cvtColor(frame_to_use, cv2.COLOR_BGR2GRAY)
                    frame_redimensionado = cv2.resize(frame_cinza, tamanho_imagem)
                    frames_selecionados.append(frame_redimensionado)
                else:
                    if frames_selecionados:
                        frames_selecionados.append(frames_selecionados[-1])
                    else:
                        print(f"Aviso: Frame {idx_target} não encontrado em {nome_arquivo}")
                        break


            while len(frames_selecionados) < num_frames_por_video:
                frames_selecionados.append(frames_selecionados[-1])

            todos_frames.extend(frames_selecionados[:num_frames_por_video])
            todos_rotulos.extend([rotulo_classe] * num_frames_por_video)

    print(f"Carregamento concluído. Total de frames: {len(todos_frames)}")
    return np.array(todos_frames), np.array(todos_rotulos), nomes_classes

# ==================== Extrator HOG  ====================
class HOGFeatureExtractor:
    @staticmethod
    def calcularGradiente(imagem):
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
                img = cv2.resize(img, tamanho_imagem)
            img = img.astype(np.float32)

            magnitude, angulo = HOGFeatureExtractor.calcularGradiente(img)

            altura, largura = img.shape
            celula_x = largura // tamanho_celula
            celula_y = altura // tamanho_celula

            # Verifica se os tamanhos são divisíveis
            if largura % tamanho_celula != 0 or altura % tamanho_celula != 0:
                 print(f"Erro HOG: Tamanho da imagem ({largura},{altura}) não é divisível pelo tamanho da célula ({tamanho_celula}). Pulando imagem.")
                 est_blocos_y = celula_y - tamanho_bloco + 1
                 est_blocos_x = celula_x - tamanho_bloco + 1
                 tamanho_esperado = est_blocos_y * est_blocos_x * tamanho_bloco * tamanho_bloco * bins
                 hog_vetores.append(np.zeros(tamanho_esperado)) 
                 continue

            histograma_orientacoes = np.zeros((celula_y, celula_x, bins))

            for i in range(celula_y):
                for j in range(celula_x):
                    mag_celula = magnitude[i*tamanho_celula:(i+1)*tamanho_celula, j*tamanho_celula:(j+1)*tamanho_celula]
                    ang_celula = angulo[i*tamanho_celula:(i+1)*tamanho_celula, j*tamanho_celula:(j+1)*tamanho_celula]

                    hist = np.zeros(bins)
                    for y in range(tamanho_celula):
                        for x in range(tamanho_celula):
                            # Evita divisão por zero se bins for 1
                            bin_idx = int(ang_celula[y, x] / (180.0 / bins)) % bins if bins > 0 else 0
                            hist[bin_idx] += mag_celula[y, x]
                    histograma_orientacoes[i, j] = hist

            blocos_y = celula_y - tamanho_bloco + 1
            blocos_x = celula_x - tamanho_bloco + 1
            blocos_normalizados = []

            for y in range(blocos_y):
                for x in range(blocos_x):
                    bloco = histograma_orientacoes[y:y+tamanho_bloco, x:x+tamanho_bloco].flatten()
                    # Normalização L2-Hys (L2 seguido de clipping e L2 novamente)
                    norm = np.linalg.norm(bloco) + 1e-6 # Evita divisão por zero
                    bloco_normalizado = bloco / norm
                    bloco_normalizado = np.minimum(bloco_normalizado, 0.2) # Clipping
                    norm = np.linalg.norm(bloco_normalizado) + 1e-6 # Renormaliza
                    bloco_normalizado = bloco_normalizado / norm
                    blocos_normalizados.append(bloco_normalizado)

            vetor_hog = np.concatenate(blocos_normalizados) if blocos_normalizados else np.array([])
            hog_vetores.append(vetor_hog)

        if hog_vetores:
            tamanho_ref = len(hog_vetores[0])
            for i, v in enumerate(hog_vetores):
                if len(v) != tamanho_ref:
                    print(f"Aviso HOG: Vetor {i} tem tamanho {len(v)}, esperado {tamanho_ref}. Preenchendo com zeros.")
                    vetor_ajustado = np.zeros(tamanho_ref)
                    len_min = min(len(v), tamanho_ref)
                    vetor_ajustado[:len_min] = v[:len_min]
                    hog_vetores[i] = vetor_ajustado

        hog_vetores_np = np.array(hog_vetores)
        print(f"Shape features HOG (custom): {hog_vetores_np.shape}")
        return hog_vetores_np

# ==================== Extrator CNN (MobileNetV2) ====================
mobilenet = None
def inicializar_mobilenet(tamanho_imagem):
    global mobilenet
    if mobilenet is None:
        mobilenet = MobileNetV2(weights='imagenet', include_top=False, pooling='avg',
                                input_shape=(tamanho_imagem[1], tamanho_imagem[0], 3))
        mobilenet.trainable = False
        print(f"MobileNetV2 inicializado com input_shape={(tamanho_imagem[1], tamanho_imagem[0], 3)}")

def extrair_cnn(imagens_gray):
    global mobilenet
    if mobilenet is None:
        raise RuntimeError("MobileNetV2 não inicializado.")

    if imagens_gray.ndim == 3:
        imagens_gray = np.expand_dims(imagens_gray, axis=-1)
    if imagens_gray.shape[-1] == 1:
        imgs_rgb = np.repeat(imagens_gray, 3, axis=-1)
    elif imagens_gray.shape[-1] == 3:
        imgs_rgb = imagens_gray
    else:
        raise ValueError(f"Formato de imagem inesperado: {imagens_gray.shape}")

    imgs_rgb = imgs_rgb.astype(np.float32)
    imgs_rgb = preprocess_input(imgs_rgb)

    print(f"Extraindo features CNN para {len(imgs_rgb)} frames...")
    extracao = mobilenet.predict(imgs_rgb, batch_size=32, verbose=0)
    print(f"Shape features CNN: {extracao.shape}")
    return extracao

# ==================== Preparar Dados para LSTM ====================
def preparar_dados_video_LSTM(frames_cinza, rotulos, seq_length, tamanho_imagem=(48, 48)):

    num_total_frames = frames_cinza.shape[0]
    if num_total_frames == 0:
        print("Erro: Nenhum frame fornecido.")
        return None, None

    if num_total_frames % seq_length!= 0:
        print(f"Aviso: Número total de frames ({num_total_frames}) não é divisível por tamanho_sequencia({seq_length}).")
        num_videos = num_total_frames // seq_length
        if num_videos == 0:
             print(f"Erro: Menos frames ({num_total_frames}) que tamanho_sequencia({seq_length}).")
             return None, None
        frames_cinza = frames_cinza[:num_videos * seq_length]
        rotulos = rotulos[:num_videos * seq_length]
        print(f"Usando {num_videos * seq_length} frames para formar {num_videos} sequências.")
    else:
        num_videos = num_total_frames // seq_length

    print(f"Preparando {num_videos} sequências de {seq_length} frames cada.")


    features_hog = HOGFeatureExtractor.calcularHistogramaCelula(frames_cinza, tamanho_imagem=tamanho_imagem)

    try:
        features_cnn = extrair_cnn(frames_cinza)
    except Exception as e:
        print(f"Erro durante a extração de features CNN: {e}")
        return None, None

    print(f"Concatenando HOG ({features_hog.shape}) e CNN ({features_cnn.shape}) features...")
    try:
        features_combinadas = np.concatenate([features_hog, features_cnn], axis=1)
    except ValueError as e:
        print(f"Erro ao concatenar features: {e}. Shapes: HOG={features_hog.shape}, CNN={features_cnn.shape}")
        return None, None
    print(f"Shape features combinadas: {features_combinadas.shape}")

    # --- Remodelar para Sequências --- 
    num_features_combinadas = features_combinadas.shape[1]
    try:
        X_seq = features_combinadas.reshape((num_videos, seq_length, num_features_combinadas))
    except ValueError as e:
        print(f"Erro ao remodelar features combinadas para sequências: {e}")
        return None, None

    # --- Obter Rótulos para Sequências --- 
    y_seq = rotulos[::seq_length]

    print(f"Shape final X_seq: {X_seq.shape}")
    print(f"Shape final y_seq: {y_seq.shape}")

    return X_seq, y_seq

# ==================== Funções de Visualização e Avaliação ====================
def visualizar_resultados(historico, matriz_confusao, classes_emocao):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(historico.history['accuracy'], label='Train Accuracy')
    plt.plot(historico.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy do classificador de Emoção')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historico.history['loss'], label='Train Loss')
    plt.plot(historico.history['val_loss'], label='Validation Loss')
    plt.title('Loss do classificador de Emoção')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

    plt.figure(figsize=(12, 10))
    matriz_confusao_norm = matriz_confusao.astype('float') / matriz_confusao.sum(axis=1)[:, np.newaxis]
    sns.heatmap(matriz_confusao_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes_emocao, yticklabels=classes_emocao)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Matriz de Confusão Normalizada')
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualizar_amostra_video(X_seq_original_frames, true_labels, predicted_labels, classes_emocao):
    nome_arquivo = f'amostra_predicoes_video.png'

    plt.figure(figsize=(15, 5))
    num_amostras = min(5, X_seq_original_frames.shape[0])
    for i in range(num_amostras):
        ultimo_frame = X_seq_original_frames[i, -1, :, :]
        plt.subplot(1, num_amostras, i+1)
        plt.imshow(ultimo_frame, cmap='gray')
        try:
            true_label_idx = int(true_labels[i])
            pred_label_idx = int(predicted_labels[i])
            titulo = f"True: {classes_emocao[true_label_idx]}\nPred: {classes_emocao[pred_label_idx]}"
        except (IndexError, ValueError) as e:
            titulo = f"Erro Rótulo ({e})"
        plt.title(titulo)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(nome_arquivo)
    print(f"Visualização de amostra salva em '{nome_arquivo}'")
    plt.close()
    
def evolucao_modelo_video(model, X_teste_seq, y_teste_seq, X_teste_frames_orig, classes, seq_length, history):
    print("\nIniciando avaliação final do modelo...")
    metricas = {'accuracy': 0, 'precision': 0, 'f1': 0}
    matriz_confusao = None
    y_pred = None

    try:
        y_pred_prob = model.predict(X_teste_seq, batch_size=16, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)

        X_teste_frames_para_visualizar = None
        if X_teste_frames_orig is not None and X_teste_frames_orig.size > 0:
            num_videos_teste = X_teste_seq.shape[0]
            expected_frames = num_videos_teste * seq_length
            if X_teste_frames_orig.shape[0] >= expected_frames:
                frames_orig_teste_usados = X_teste_frames_orig[:expected_frames]
                try:
                    X_teste_frames_para_visualizar = frames_orig_teste_usados.reshape((num_videos_teste, seq_length,
                                                                              frames_orig_teste_usados.shape[1],
                                                                              frames_orig_teste_usados.shape[2]))
                except ValueError as e:
                    print(f"Aviso: Erro ao remodelar frames originais para visualização: {e}")
            else:
                print(f"Aviso: Número de frames originais ({X_teste_frames_orig.shape[0]}) menor que esperado ({expected_frames}).")

        if y_pred is not None:
             visualizar_amostra_video(X_teste_frames_para_visualizar, y_teste_seq, y_pred, classes)
        else:
             print("Predições não disponíveis para visualização de amostra.")

        if y_pred is not None:
            metricas['accuracy'] = accuracy_score(y_teste_seq, y_pred)
            metricas['precision'] = precision_score(y_teste_seq, y_pred, average='weighted', zero_division=0)
            metricas['f1'] = f1_score(y_teste_seq, y_pred, average='weighted', zero_division=0)

            labels_range = np.arange(len(classes))
            matriz_confusao = confusion_matrix(y_teste_seq, y_pred, labels=labels_range)

            print("\n=== Métricas de Avaliação Final ===")
            print(f"Accuracy: {metricas['accuracy']:.4f}")
            print(f"Precision (Weighted): {metricas['precision']:.4f}")
            print(f"F1-Score (Weighted): {metricas['f1']:.4f}")
        else:
            print("\nPredições não disponíveis, métricas não calculadas.")

    except Exception as e:
        print(f"Erro durante a avaliação do modelo: {e}")

    visualizar_resultados(history, matriz_confusao, classes)

    return metricas, matriz_confusao

# ==================== MAIN ====================
def main():
    configurar_ambiente()
    # Parâmetros
    diretorio_videos = 'S4 Stimuli (Video-clips)'
    tamanho_sequencia= 20
    largura, altura= 48, 48
    tamanho_amostra_teste = 0.2
    epocas = 100
    tamanho_batch = 16

    tamanho_imagem = (largura, altura)

    inicializar_mobilenet(tamanho_imagem)

    todos_frames_cinza, todos_rotulos, classes_emocao = carregar_frames_videos(
        diretorio_videos,
        num_frames_por_video=tamanho_sequencia,
        tamanho_imagem=tamanho_imagem
    )

    if todos_frames_cinza.size == 0: return
    num_classes = len(classes_emocao)
    if num_classes == 0: return
    print(f"Classes encontradas ({num_classes}): {classes_emocao}")
    rotulos_unicos, contagem_frames = np.unique(todos_rotulos, return_counts=True)
    print(f"Distribuição inicial (por frame): {dict(zip([classes_emocao[i] for i in rotulos_unicos], contagem_frames))}")

    # --- Divisão Treino/Teste ---
    num_total_frames = todos_frames_cinza.shape[0]
    if num_total_frames < tamanho_sequencia: return
    
    num_videos = num_total_frames // tamanho_sequencia
    print(f"Número total de vídeos: {num_videos}")
    
    indices_videos = np.arange(num_videos)
    rotulos_videos = todos_rotulos[::tamanho_sequencia]
    rotulos_vid_unicos, contagem_vids = np.unique(rotulos_videos, return_counts=True)
    print(f"Distribuição (por vídeo): {dict(zip([classes_emocao[i] for i in rotulos_vid_unicos], contagem_vids))}")

    indices_treino, indices_teste = [], []
    
    try:
        counts = np.bincount(rotulos_videos)
        if np.any(counts < 2):
            print("Aviso: Classes com < 2 amostras. Usando divisão não estratificada.")
            
            
            from collections import defaultdict
            class_to_indices = defaultdict(list)
            for idx, r in enumerate(rotulos_videos):
                class_to_indices[r].append(idx)
            
            for r, indices in class_to_indices.items():
                if len(indices) == 1:
                    indices_treino.extend(indices)
                    
                else:
                    
                    split = int(np.ceil(len(indices) * (1 - tamanho_amostra_teste))) 
                    indices_treino.extend(indices[:split])
                    indices_teste.extend(indices[:split])
                        
            
            indices_treino = list(set(indices_treino))
            indices_teste = list(set(indices_treino))
        else:
            indices_treino, indices_teste = train_test_split(
                  indices_videos, test_size=tamanho_amostra_teste, random_state=42, stratify=rotulos_videos)
    except ValueError as e:
        print(f"Erro na divisão treino/teste: {e}.")
        return
    print(f"Vídeos para treino: {len(indices_treino)}, Vídeos para teste: {len(indices_teste)}")


    def get_frame_indices(video_indices, seq_length):
        return [idx for vid_idx in video_indices for idx in range(vid_idx * seq_length, (vid_idx + 1) * seq_length)]

    frame_indices_treino = get_frame_indices(indices_treino, tamanho_sequencia)
    frame_indices_teste = get_frame_indices(indices_teste, tamanho_sequencia)

    # --- Preparar Dados Sequenciais --- 
    print("\nPreparando dados de TREINO...")
    frames_treino = todos_frames_cinza[frame_indices_treino]
    rotulos_treino = todos_rotulos[frame_indices_treino]
    X_treino_seq, y_treino_seq = preparar_dados_video_LSTM(frames_treino, rotulos_treino, tamanho_sequencia, tamanho_imagem)

    print("\nPreparando dados de TESTE...")
    frames_teste = todos_frames_cinza[frame_indices_teste]
    rotulos_teste = todos_rotulos[frame_indices_teste]
    X_teste_frames_orig = frames_teste.copy()
    X_teste_seq, y_teste_seq = preparar_dados_video_LSTM(frames_teste, rotulos_teste, tamanho_sequencia, tamanho_imagem)

    if X_treino_seq is None or y_treino_seq is None or X_teste_seq is None or y_teste_seq is None: return
    if X_treino_seq.shape[0] == 0 or X_teste_seq.shape[0] == 0: return
    print(f"\nDados prontos: Treino X={X_treino_seq.shape}, y={y_treino_seq.shape}; Teste X={X_teste_seq.shape}, y={y_teste_seq.shape}")

    # --- Construção do Modelo LSTM --- 
    input_shape_lstm = (X_treino_seq.shape[1], X_treino_seq.shape[2])
    print(f"Input shape para LSTM: {input_shape_lstm}")

    classificador = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape_lstm),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-7, verbose=1),
        callbacks.ModelCheckpoint(f'best_video_lstm_model.keras', monitor='val_accuracy', save_best_only=True, verbose=0)
    ]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    classificador.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    classificador.summary()

    # --- Treinamento --- 
    print(f"\nIniciando o treinamento do modelo ...")
    history = None
    try:
        history = classificador.fit(
            X_treino_seq, y_treino_seq,
            validation_data=(X_teste_seq, y_teste_seq),
            epochs=epocas,
            batch_size=tamanho_batch,
            callbacks=callbacks_list,
            verbose=1
        )
        print("Treinamento finalizado.")
        classificador.save(f'emotion_lstm_final.h5')
        np.save(f'emotion_video_classes.npy', np.array(classes_emocao))
        print(f"Modelo final e classes salvos.")

    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        try:
            print("Tentando carregar o melhor modelo salvo...")
            classificador = tf.keras.models.load_model(f'best_video_lstm_model.keras')
            print("Melhor modelo carregado.")
        except Exception as load_e:
            print(f"Falha ao carregar o melhor modelo: {load_e}")
            classificador = None

    # --- Avaliação Final --- 
    if classificador is not None:
        evolucao_modelo_video(
            classificador, X_teste_seq, y_teste_seq, X_teste_frames_orig,
            classes_emocao, tamanho_sequencia, history
        )

    print(f"\nScript concluído.")

if __name__ == "__main__":
    main()



