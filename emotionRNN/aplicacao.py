import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import StandardScaler
import os
import sys
import argparse
import json
import pickle
from pathlib import Path
from tqdm import tqdm
from collections import Counter, deque
import matplotlib.pyplot as plt
import seaborn as sns

class HOGFeatureExtractor:    
    @staticmethod
    def calcularGradiente(imagem):
        imagem = cv2.GaussianBlur(imagem.astype(np.float32), (3, 3), 0)
        gradienteX = cv2.Sobel(imagem, cv2.CV_32F, 1, 0, ksize=3)
        gradienteY = cv2.Sobel(imagem, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(gradienteX**2 + gradienteY**2)
        angulo = np.arctan2(gradienteY, gradienteX) * (180 / np.pi)
        angulo[angulo < 0] += 180
        return magnitude, angulo

    @staticmethod
    def calcularHistogramaCelula(imagens, tamanho_imagem=(48,48), tamanho_bloco=2, tamanho_celula=6, bins=12):
        hog_vetores = []
        
        print(f"Calculando HOG para {len(imagens)} frames...")
        for img in tqdm(imagens, desc="Enhanced HOG Features"):
            if img.shape != tamanho_imagem:
                img = cv2.resize(img, tamanho_imagem, interpolation=cv2.INTER_AREA)
            
            img = img.astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255

            magnitude, angulo = HOGFeatureExtractor.calcularGradiente(img)

            altura, largura = img.shape
            celula_x = largura // tamanho_celula
            celula_y = altura // tamanho_celula

            histograma_orientacoes = np.zeros((celula_y, celula_x, bins))

            for i in range(celula_y):
                for j in range(celula_x):
                    mag_celula = magnitude[i*tamanho_celula:(i+1)*tamanho_celula, 
                                         j*tamanho_celula:(j+1)*tamanho_celula]
                    ang_celula = angulo[i*tamanho_celula:(i+1)*tamanho_celula, 
                                       j*tamanho_celula:(j+1)*tamanho_celula]

                    hist = np.zeros(bins)
                    for y_cell in range(tamanho_celula):
                        for x_cell in range(tamanho_celula):
                            if y_cell < mag_celula.shape[0] and x_cell < mag_celula.shape[1]:
                                angle_val = ang_celula[y_cell, x_cell]
                                mag_val = mag_celula[y_cell, x_cell]
                                
                                bin_idx = angle_val / (180.0 / bins)
                                bin_low = int(bin_idx) % bins
                                bin_high = (bin_low + 1) % bins
                                
                                weight_high = bin_idx - int(bin_idx)
                                weight_low = 1.0 - weight_high
                                
                                hist[bin_low] += mag_val * weight_low
                                hist[bin_high] += mag_val * weight_high
                    
                    histograma_orientacoes[i, j] = hist

            blocos_y = celula_y - tamanho_bloco + 1
            blocos_x = celula_x - tamanho_bloco + 1
            blocos_normalizados = []

            for y_block in range(blocos_y):
                for x_block in range(blocos_x):
                    bloco = histograma_orientacoes[y_block:y_block+tamanho_bloco, 
                                                 x_block:x_block+tamanho_bloco].flatten()
                    
                    norm = np.linalg.norm(bloco) + 1e-6
                    bloco_normalizado = bloco / norm
                    
                    bloco_normalizado = np.minimum(bloco_normalizado, 0.2)
                    
                    norm = np.linalg.norm(bloco_normalizado) + 1e-6
                    bloco_normalizado = bloco_normalizado / norm
                    
                    blocos_normalizados.append(bloco_normalizado)

            vetor_hog = np.concatenate(blocos_normalizados) if blocos_normalizados else np.array([])
            hog_vetores.append(vetor_hog)

        if hog_vetores:
            hog_vetores_np = np.array(hog_vetores)
        else:
            hog_vetores_np = np.array([])

        print(f"HOG features shape: {hog_vetores_np.shape}")
        return hog_vetores_np

class EmotionVideoDetector:
    def __init__(self, model_path, scalers_path=None):

        print("Inicializando detector ...")
        
        self.model = load_model(model_path)
        print(f"Modelo carregado: {model_path}")
        
        self.classes = np.array(['Feliz', 'Raiva', 'Medo', 'Triste', 'Desgosto', 'Surpresa'])
        
        self.mobilenet = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            pooling='avg', 
            input_shape=(48,48, 3) 
        )
        self.mobilenet.trainable = True
        for layer in self.mobilenet.layers[:-20]:
            layer.trainable = False
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        self.emotion_colors = {
            'Feliz': (0, 255, 0),      # Verde
            'Raiva': (0, 0, 255),      # Vermelho
            'Medo': (128, 0, 128),     # Roxo
            'Triste': (255, 0, 0),     # Azul
            'Desgosto': (0, 128, 128), # Marrom
            'Surpresa': (0, 255, 255)  # Amarelo
        }
        
        self.tamanho_imagem = (48,48)  
        self.seq_length = 20
        
        self.hog_scaler = StandardScaler()
        self.cnn_scaler = StandardScaler()
        self.scalers_fitted = False
        
        if scalers_path and os.path.exists(scalers_path):
            self.carregar_scalers(scalers_path)
        
        self.emotion_buffer = deque(maxlen=5)
        self.confidence_buffer = deque(maxlen=5)
        
        self.frame_count = 0
        self.emotion_history = []

    
    def carregar_scalers(self, scalers_path):
        try:
            with open(scalers_path, 'rb') as f:
                scalers_data = pickle.load(f)
                self.hog_scaler = scalers_data['hog_scaler']
                self.cnn_scaler = scalers_data['cnn_scaler']
                self.scalers_fitted = True
                print(f"Scalers carregados de: {scalers_path}")
        except Exception as e:
            print(f"Aviso: Não foi possível carregar scalers: {e}")
            print("Scalers serão ajustados durante o processamento")
    
    def salvar_scalers(self, scalers_path):
        try:
            scalers_data = {
                'hog_scaler': self.hog_scaler,
                'cnn_scaler': self.cnn_scaler
            }
            with open(scalers_path, 'wb') as f:
                pickle.dump(scalers_data, f)
            print(f"Scalers salvos em: {scalers_path}")
        except Exception as e:
            print(f"Erro ao salvar scalers: {e}")
    
    def processar_frame_individual(self, frame):
        frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_cinza = cv2.equalizeHist(frame_cinza)
        
        faces = self.face_cascade.detectMultiScale(
            frame_cinza, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) > 0:
            areas = [w * h for (x, y, w, h) in faces]
            max_area_idx = np.argmax(areas)
            (x, y, w, h) = faces[max_area_idx]
            
            padding = int(0.1 * min(w, h))
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame_cinza.shape[1] - x, w + 2 * padding)
            h = min(frame_cinza.shape[0] - y, h + 2 * padding)
            
            face_roi = frame_cinza[y:y+h, x:x+w]
            if face_roi.size > 0:
                frame_processado = cv2.resize(face_roi, self.tamanho_imagem, interpolation=cv2.INTER_AREA)
            else:
                frame_processado = cv2.resize(frame_cinza, self.tamanho_imagem, interpolation=cv2.INTER_AREA)
        else:
            frame_processado = cv2.resize(frame_cinza, self.tamanho_imagem, interpolation=cv2.INTER_AREA)
        
        return frame_processado, faces
    
    def extrair_todos_frames(self, video_path):
        print(f"Extraindo frames de: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frames_originais = []
        frames_processados = []
        faces_detectadas = []
        
        print(f"Processando {total_frames} frames (FPS: {fps:.1f})...")
        
        frame_count = 0
        with tqdm(total=total_frames, desc="Extraindo frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_originais.append(frame)
                frame_processado, faces = self.processar_frame_individual(frame)
                frames_processados.append(frame_processado)
                faces_detectadas.append(faces)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        self.frame_count = frame_count
        
        print(f"Extraídos {len(frames_processados)} frames processados")
        return np.array(frames_processados), frames_originais, faces_detectadas
    
    def extrair_cnn_features(self, frames_gray):
        print(f"Extraindo CNN features melhoradas para {len(frames_gray)} frames...")
        
        if frames_gray.ndim == 3:
            frames_gray = np.expand_dims(frames_gray, axis=-1)
        
        # Conversão para RGB
        imgs_rgb = np.repeat(frames_gray, 3, axis=-1)
        imgs_rgb = imgs_rgb.astype(np.float32)
        
        # Normalização melhorada (igual ao treinamento)
        imgs_rgb = (imgs_rgb - 127.5) / 127.5  # Normalização para [-1, 1]
        imgs_rgb = preprocess_input(imgs_rgb)
        
        # Extração em batches
        batch_size = 16  # Batch menor para melhor performance
        all_features = []
        
        for i in tqdm(range(0, len(imgs_rgb), batch_size), desc="CNN Features"):
            batch = imgs_rgb[i:i+batch_size]
            features = self.mobilenet.predict(batch, verbose=0)
            all_features.append(features)
        
        cnn_features = np.vstack(all_features)
        print(f"CNN features shape: {cnn_features.shape}")
        return cnn_features
    
    def preparar_features_combinadas(self, hog_features, cnn_features):
        print("Preparando features combinadas com normalização...")
        
        if not self.scalers_fitted:
            print("Ajustando scalers aos dados...")
            self.hog_scaler.fit(hog_features)
            self.cnn_scaler.fit(cnn_features)
            self.scalers_fitted = True
        
        # Normalizar features
        hog_normalized = self.hog_scaler.transform(hog_features)
        cnn_normalized = self.cnn_scaler.transform(cnn_features)
        
        # Combinação ponderada (igual ao treinamento)
        peso_hog = 0.5
        peso_cnn = 0.5
        
        features_combinadas = np.concatenate([
            hog_normalized * peso_hog,
            cnn_normalized * peso_cnn
        ], axis=1)
        
        print(f"Features combinadas shape: {features_combinadas.shape}")
        return features_combinadas
    
    def preparar_sequencia_lstm(self, features_combinadas, start_frame):
        """Prepara sequência para o modelo LSTM"""
        end_frame = start_frame + self.seq_length
        
        if end_frame <= len(features_combinadas):
            sequencia = features_combinadas[start_frame:end_frame]
        else:
            available_frames = len(features_combinadas) - start_frame
            if available_frames > 0:
                sequencia = features_combinadas[start_frame:]
                padding_needed = self.seq_length - available_frames
                
                if padding_needed > 0:
                    last_feature = features_combinadas[-1]
                    padding = np.tile(last_feature, (padding_needed, 1))
                    sequencia = np.vstack([sequencia, padding])
            else:
                return None
        
        # Reshape para entrada do LSTM
        lstm_input = sequencia.reshape((1, self.seq_length, sequencia.shape[1]))
        return lstm_input
    
    def prever_emocao_com_suavizacao(self, lstm_input):
        if lstm_input is None:
            return None, 0.0
        
        # Predição
        predicao = self.model.predict(lstm_input, verbose=0)
        classe_idx = np.argmax(predicao)
        confianca = np.max(predicao)
        classe_predita = self.classes[classe_idx]
        
        # Adicionar ao buffer para suavização
        self.emotion_buffer.append(classe_predita)
        self.confidence_buffer.append(confianca)
        
        if len(self.emotion_buffer) >= 3:
            emotion_counts = Counter(self.emotion_buffer)
            emocao_suavizada = emotion_counts.most_common(1)[0][0]
            confianca_media = np.mean(list(self.confidence_buffer))
        else:
            emocao_suavizada = classe_predita
            confianca_media = confianca
        
        return emocao_suavizada, confianca_media
    
    def desenhar_interface(self, frame, emocao, confianca, faces, frame_idx, total_frames):
        frame_result = frame.copy()
        height, width = frame_result.shape[:2]
        cor = self.emotion_colors.get(emocao, (255, 255, 255))
        
        # Desenhar retângulos nas faces detectadas
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Retângulo da face
                cv2.rectangle(frame_result, (x, y), (x+w, y+h), cor, 3)
                
                # Label da emoção
                label = f"{emocao}: {confianca:.2f}"
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                # Fundo do texto
                cv2.rectangle(frame_result, (x, y-text_height-15), (x+text_width+10, y), cor, -1)
                cv2.putText(frame_result, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Painel de informações principal
        panel_height = 10
        panel_color = (40, 40, 40)
        cv2.rectangle(frame_result, (0, 0), (width, panel_height), panel_color, -1)
        
        
        # Emoção atual
        cv2.putText(frame_result, f"Emocao: {emocao}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        
        # Confiança
        cv2.putText(frame_result, f"Confianca: {confianca:.1%}", 
                   (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        
        # Progresso
        progress = (frame_idx + 1) / total_frames
        progress_width = int(progress * (width - 20))
        cv2.rectangle(frame_result, (10, height-30), (width-10, height-20), (60, 60, 60), -1)
        cv2.rectangle(frame_result, (10, height-30), (10 + progress_width, height-20), cor, -1)
        cv2.putText(frame_result, f"Progresso: {progress:.1%} ({frame_idx+1}/{total_frames})", 
                   (10, height-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Barra de confiança lateral
        confidence_bar_height = int(confianca * 100)
        cv2.rectangle(frame_result, (width-30, height-120), (width-10, height-20), (50, 50, 50), -1)
        cv2.rectangle(frame_result, (width-30, height-20-confidence_bar_height), 
                     (width-10, height-20), cor, -1)
        
        return frame_result
    
    def processar_video_completo(self, input_path, output_path, salvar_estatisticas=True):
        print("=== PROCESSAMENTO DE VÍDEO  ===")
        
        # Extrair frames
        frames_processados, frames_originais, faces_por_frame = self.extrair_todos_frames(input_path)
        
        if len(frames_processados) == 0:
            raise ValueError("Nenhum frame foi extraído do vídeo")
        
        # Extrair features HOG 
        hog_features = HOGFeatureExtractor.calcularHistogramaCelula(
            frames_processados, 
            tamanho_imagem=self.tamanho_imagem,
            tamanho_bloco=2, 
            tamanho_celula=6, 
            bins=12  
        )
        
        # Extrair features CNN melhoradas
        cnn_features = self.extrair_cnn_features(frames_processados)
        
        # Combinar features
        features_combinadas = self.preparar_features_combinadas(hog_features, cnn_features)
        
        # Configurar vídeo de saída
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processar sequências
        total_frames = len(frames_originais)
        emocoes_detectadas = []
        confidencias = []
        
        print(f"Processando {total_frames} frames em sequências de {self.seq_length}...")
        
        emocao_atual = "Neutro"
        confianca_atual = 0.0
        
        for frame_idx in tqdm(range(total_frames), desc="Processando vídeo"):
            # Atualizar predição a cada seq_length frames
            if frame_idx % self.seq_length == 0:
                lstm_input = self.preparar_sequencia_lstm(features_combinadas, frame_idx)
                
                if lstm_input is not None:
                    emocao, confianca = self.prever_emocao_com_suavizacao(lstm_input)
                    if emocao is not None:
                        emocao_atual = emocao
                        confianca_atual = confianca
                        emocoes_detectadas.append(emocao)
                        confidencias.append(confianca)
            
            # Desenhar interface no frame
            frame = frames_originais[frame_idx]
            faces = faces_por_frame[frame_idx]
            frame_processado = self.desenhar_interface(
                frame, emocao_atual, confianca_atual, faces, frame_idx, total_frames
            )
            
            out.write(frame_processado)
            
            # Atualizar histórico
            self.emotion_history.append({
                'frame': frame_idx,
                'emotion': emocao_atual,
                'confidence': confianca_atual
            })
        
        out.release()
        
        # Calcular estatísticas
        estatisticas = self.calcular_estatisticas(emocoes_detectadas, confidencias)
        
        # Salvar estatísticas se solicitado
        if salvar_estatisticas:
            self.salvar_estatisticas(output_path, estatisticas)
            self.gerar_graficos_analise(output_path, estatisticas)
        
        print(f"\nVídeo processado salvo em: {output_path}")
        self.imprimir_estatisticas(estatisticas)
        
        return estatisticas
    
    def calcular_estatisticas(self, emocoes_detectadas, confidencias):
        """Calcula estatísticas detalhadas do processamento"""
        if not emocoes_detectadas:
            return {}
        
        contagem_emocoes = Counter(emocoes_detectadas)
        total_predicoes = len(emocoes_detectadas)
        
        estatisticas = {
            'total_frames': self.frame_count,
            'total_predicoes': total_predicoes,
            'emocao_predominante': contagem_emocoes.most_common(1)[0][0],
            'distribuicao_emocoes': dict(contagem_emocoes),
            'porcentagens': {emocao: (count/total_predicoes)*100 
                           for emocao, count in contagem_emocoes.items()},
            'confianca_media': np.mean(confidencias),
            'confianca_std': np.std(confidencias),
            'confianca_min': np.min(confidencias),
            'confianca_max': np.max(confidencias),
            'emocao_final': emocoes_detectadas[-1] if emocoes_detectadas else "Desconhecida",
            'confianca_final': confidencias[-1] if confidencias else 0.0
        }
        
        return estatisticas
    
    def imprimir_estatisticas(self, estatisticas):
        print("\n" + "="*60)
        print("ESTATÍSTICAS DO PROCESSAMENTO")
        print("="*60)
        print(f"Total de frames: {estatisticas.get('total_frames', 0)}")
        print(f"Total de predições: {estatisticas.get('total_predicoes', 0)}")
        print(f"Emoção predominante: {estatisticas.get('emocao_predominante', 'N/A')}")
        print(f"Emoção final: {estatisticas.get('emocao_final', 'N/A')}")
        print(f"Confiança final: {estatisticas.get('confianca_final', 0):.1%}")
        
        print(f"\nConfiança média: {estatisticas.get('confianca_media', 0):.3f}")
        print(f"Desvio padrão: {estatisticas.get('confianca_std', 0):.3f}")
        print(f"Confiança mín/máx: {estatisticas.get('confianca_min', 0):.3f} / {estatisticas.get('confianca_max', 0):.3f}")
        
        print("\nDistribuição de emoções:")
        for emocao, porcentagem in estatisticas.get('porcentagens', {}).items():
            count = estatisticas.get('distribuicao_emocoes', {}).get(emocao, 0)
            print(f"  {emocao}: {count} vezes ({porcentagem:.1f}%)")
        
        print("="*60)
    
    def salvar_estatisticas(self, output_path, estatisticas):
        stats_path = Path(output_path).with_suffix('.json')
        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(estatisticas, f, indent=2, ensure_ascii=False)
            print(f"Estatísticas salvas em: {stats_path}")
        except Exception as e:
            print(f"Erro ao salvar estatísticas: {e}")
    
    def gerar_graficos_analise(self, output_path, estatisticas):
        """Gera gráficos de análise"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Gráfico de pizza - distribuição de emoções
            emocoes = list(estatisticas['distribuicao_emocoes'].keys())
            counts = list(estatisticas['distribuicao_emocoes'].values())
            colors = [self.emotion_colors.get(em, (128, 128, 128)) for em in emocoes]
            colors = [(r/255, g/255, b/255) for b, g, r in colors]  # BGR to RGB
            
            axes[0, 0].pie(counts, labels=emocoes, autopct='%1.1f%%', colors=colors)
            axes[0, 0].set_title('Distribuição de Emoções')
            
            # Gráfico de barras - contagem de emoções
            axes[0, 1].bar(emocoes, counts, color=colors)
            axes[0, 1].set_title('Contagem de Emoções')
            axes[0, 1].set_ylabel('Número de Detecções')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Linha temporal das emoções (se disponível)
            if hasattr(self, 'emotion_history') and self.emotion_history:
                frames = [h['frame'] for h in self.emotion_history[::10]]  # Amostragem
                emotions_numeric = [list(self.classes).index(h['emotion']) 
                                  for h in self.emotion_history[::10]]
                
                axes[1, 0].plot(frames, emotions_numeric, marker='o', markersize=2)
                axes[1, 0].set_title('Evolução Temporal das Emoções')
                axes[1, 0].set_xlabel('Frame')
                axes[1, 0].set_ylabel('Emoção')
                axes[1, 0].set_yticks(range(len(self.classes)))
                axes[1, 0].set_yticklabels(self.classes)
                axes[1, 0].grid(True, alpha=0.3)
            
            # Histograma de confiança
            if hasattr(self, 'emotion_history') and self.emotion_history:
                confidences = [h['confidence'] for h in self.emotion_history]
                axes[1, 1].hist(confidences, bins=20, alpha=0.7, color='skyblue')
                axes[1, 1].set_title('Distribuição de Confiança')
                axes[1, 1].set_xlabel('Confiança')
                axes[1, 1].set_ylabel('Frequência')
                axes[1, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                                 label=f'Média: {np.mean(confidences):.3f}')
                axes[1, 1].legend()
            
            plt.tight_layout()
            
            # Salvar gráfico
            graph_path = Path(output_path).with_suffix('.png')
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gráficos de análise salvos em: {graph_path}")
            
        except Exception as e:
            print(f"Erro ao gerar gráficos: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Detector de Emoções",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_video", help="Vídeo de entrada para processamento")
    parser.add_argument("-o", "--output", help="Vídeo de saída (padrão: <entrada>_emotions.mp4)", default=None)
    parser.add_argument("-m", "--model", help="Modelo LSTM  ", 
                       default="classificador_emocaoLS.keras")
    parser.add_argument("-s", "--scalers", help="Arquivo de scalers pré-treinados", default=None)
    parser.add_argument("--save-scalers", help="Salvar scalers treinados para uso futuro", action="store_true")
    parser.add_argument("--no-stats", help="Não salvar estatísticas e gráficos", action="store_true")
    parser.add_argument("--verbose", help="Modo verboso", action="store_true")
    
    args = parser.parse_args()
    
    if not args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    if not os.path.exists(args.input_video):
        print(f"❌ Erro: Vídeo não encontrado: {args.input_video}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Erro: Modelo não encontrado: {args.model}")
        print("💡 Dica: Execute primeiro o treinamento para gerar o modelo")
        sys.exit(1)
    
    if args.output is None:
        input_path = Path(args.input_video)
        output_path = input_path.parent / f"{input_path.stem}{input_path.suffix}"
    else:
        output_path = Path(args.output)
    
    try:
        print("=== DETECTOR DE EMOÇÕES ===")
        
        detector = EmotionVideoDetector(args.model, args.scalers)
        
        estatisticas = detector.processar_video_completo(
            args.input_video, 
            str(output_path),
            salvar_estatisticas=not args.no_stats
        )
        
        if args.save_scalers and detector.scalers_fitted:
            scalers_path = output_path.with_suffix('.scalers.pkl')
            detector.salvar_scalers(str(scalers_path))
        
        print(f"\nProcessamento concluído com sucesso!")
        print(f"Vídeo salvo: {output_path}")
        print(f"Emoção predominante: {estatisticas.get('emocao_predominante', 'N/A')}")
        print(f"Confiança média: {estatisticas.get('confianca_media', 0):.1%}")
        
        if not args.no_stats:
            print(f"Estatísticas: {output_path.with_suffix('.json')}")
            print(f"Gráficos: {output_path.with_suffix('.png')}")
        
    except KeyboardInterrupt:
        print("\nProcessamento interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"Erro durante o processamento: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

