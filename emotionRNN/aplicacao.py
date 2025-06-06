import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

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
        """Calcula HOG para múltiplas imagens de uma vez"""
        hog_vetores = []
        
        print(f"Calculando HOG para {len(imagens)} frames...")
        for img in tqdm(imagens, desc="HOG Features"):
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
                            if y_cell < mag_celula.shape[0] and x_cell < mag_celula.shape[1]:
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
                    bloco_normalizado = np.minimum(bloco_normalizado, 0.2)
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
                    vetor_ajustado = np.zeros(tamanho_ref)
                    len_min = min(len(v), tamanho_ref)
                    vetor_ajustado[:len_min] = v[:len_min]
                    hog_vetores_consistentes.append(vetor_ajustado)
            hog_vetores_np = np.array(hog_vetores_consistentes)
        else:
            hog_vetores_np = np.array([])

        print(f"HOG features shape: {hog_vetores_np.shape}")
        return hog_vetores_np

class EmotionVideoDetector:
    def __init__(self, model_path):       
        self.model = load_model(model_path)
        
        self.classes = np.array(['Feliz','Raiva', 'Medo', 'Triste','Desgosto', 'Surpresa'])        
        
        self.mobilenet = tf.keras.applications.MobileNetV2(
            weights='imagenet', include_top=False, pooling='avg', input_shape=(48, 48, 3)
        )
        self.mobilenet.trainable = False
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        self.emotion_colors = {
            'Feliz': (0, 255, 0),
            'Raiva': (0, 0, 255),
            'Medo': (128, 0, 128),
            'Triste': (255, 0, 0),
            'Desgosto': (0, 128, 0),
            'Surpresa': (0, 255, 255)
        }
        
        self.hog_cache = {}
        self.cnn_cache = {}
        
        print(f"Modelo carregado com {len(self.classes)} classes: {self.classes}")
        
    def extrair_todos_frames(self, video_path):
        print(f"Extraindo todos os frames de: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_originais = []
        frames_processados = []
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frames_originais.append(frame)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            frames_processados.append(resized)
            
            frame_count += 1
        
        cap.release()
        
        print(f"Extraídos {len(frames_processados)} frames processados")
        return np.array(frames_processados), frames_originais
    
    def extrair_cnn_todas_features(self, frames_gray):
        """Extrai features CNN para todos os frames de uma vez"""
        print(f"Extraindo CNN features para {len(frames_gray)} frames...")
        
        if frames_gray.ndim == 3:
            frames_gray = np.expand_dims(frames_gray, axis=-1)
        
        imgs_rgb = np.repeat(frames_gray, 3, axis=-1)
        imgs_rgb = imgs_rgb.astype(np.float32)
        imgs_rgb = preprocess_input(imgs_rgb)
        
        batch_size = 32
        all_features = []
        
        for i in range(0, len(imgs_rgb), batch_size):
            batch = imgs_rgb[i:i+batch_size]
            features = self.mobilenet.predict(batch, verbose=0)
            all_features.append(features)
        
        cnn_features = np.vstack(all_features)
        print(f"CNN features shape: {cnn_features.shape}")
        return cnn_features
    
    def preparar_sequencia_lstm(self, hog_features, cnn_features, start_frame, seq_length=20):
        end_frame = start_frame + seq_length
        
        if end_frame <= len(hog_features):
            hog_seq = hog_features[start_frame:end_frame]
            cnn_seq = cnn_features[start_frame:end_frame]
        else:
            available_frames = len(hog_features) - start_frame
            if available_frames > 0:
                hog_seq = hog_features[start_frame:]
                cnn_seq = cnn_features[start_frame:]
                
                padding_needed = seq_length - available_frames
                if padding_needed > 0:
                    last_hog = hog_features[-1]
                    last_cnn = cnn_features[-1]
                    
                    hog_padding = np.tile(last_hog, (padding_needed, 1))
                    cnn_padding = np.tile(last_cnn, (padding_needed, 1))
                    
                    hog_seq = np.vstack([hog_seq, hog_padding])
                    cnn_seq = np.vstack([cnn_seq, cnn_padding])
            else:
                return None
        
        combined_features = np.concatenate([hog_seq, cnn_seq], axis=1)
        
        lstm_input = combined_features.reshape((1, seq_length, combined_features.shape[1]))
        
        return lstm_input
    
    def prever_emocao_sequencia(self, lstm_input):
        if lstm_input is None:
            return None, 0.0
        
        predicao = self.model.predict(lstm_input, verbose=0)
        classe_idx = np.argmax(predicao)
        confianca = np.max(predicao)
        classe_predita = self.classes[classe_idx]
        
        return classe_predita, confianca
    
    def detectar_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def desenhar_emocao_no_frame(self, frame, emocao, confianca, faces):
        frame_result = frame.copy()
        cor = self.emotion_colors.get(emocao, (255, 255, 255))
        
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame_result, (x, y), (x+w, y+h), cor, 3)
                label = f"{emocao}: {confianca:.2f}"
                
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame_result, (x, y-text_height-15), (x+text_width+10, y), cor, -1)
                cv2.putText(frame_result, label, (x+5, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            label = f"Emocao: {emocao} ({confianca:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(frame_result, (10, 10), (text_width+20, text_height+20), cor, -1)
            cv2.putText(frame_result, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Info do pipeline
        height, width = frame_result.shape[:2]
        cv2.putText(frame_result, "Pipeline: Haar + HOG + CNN + LSTM", 
                   (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Barra de confiança
        confidence_bar_width = int(confianca * 200)
        cv2.rectangle(frame_result, (10, height-20), (210, height-10), (50, 50, 50), -1)
        cv2.rectangle(frame_result, (10, height-20), (10 + confidence_bar_width, height-10), cor, -1)
        cv2.putText(frame_result, f"Conf: {confianca:.1%}", 
                   (220, height-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame_result
    
    def processar_video(self, input_path, output_path):

        frames_processados, frames_originais = self.extrair_todos_frames(input_path)
        
        hog_features = HOGFeatureExtractor.calcularHistogramaCelula(
            frames_processados, tamanho_imagem=(48, 48), 
            tamanho_bloco=2, tamanho_celula=6, bins=12
        )
        
        cnn_features = self.extrair_cnn_todas_features(frames_processados)
        
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        seq_length = 20
        emocao_atual = "neutral"
        confianca_atual = 0.0
        emocoes_detectadas = []
        
        total_frames = len(frames_originais)
        print(f"Processando {total_frames} frames...")
        
        for frame_idx in range(total_frames):
            if frame_idx % seq_length == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"Processando sequência {frame_idx//seq_length + 1} - Frame {frame_idx}/{total_frames} ({progress:.1f}%)")
                
                lstm_input = self.preparar_sequencia_lstm(hog_features, cnn_features, frame_idx, seq_length)
                
                if lstm_input is not None:
                    emocao, confianca = self.prever_emocao_sequencia(lstm_input)
                    if emocao is not None:
                        emocao_atual = emocao
                        confianca_atual = confianca
                        emocoes_detectadas.append(emocao)
                        print(f"  Emoção detectada: {emocao_atual} (confiança: {confianca_atual:.3f})")
            
            # Detectar faces e desenhar
            frame = frames_originais[frame_idx]
            faces = self.detectar_faces(frame)
            frame_processado = self.desenhar_emocao_no_frame(frame, emocao_atual, confianca_atual, faces)
            
            out.write(frame_processado)
        
        out.release()
        
        # Estatísticas finais
        if emocoes_detectadas:
            from collections import Counter
            contagem_emocoes = Counter(emocoes_detectadas)
            emocao_predominante = contagem_emocoes.most_common(1)[0][0]
            
            print(f"\nVídeo processado salvo em: {output_path}")
            print(f"Emoção predominante: {emocao_predominante}")
            print(f"Emoção final: {emocao_atual} (confiança: {confianca_atual:.3f})")
            print("\nDistribuição de emoções:")
            for emocao, count in contagem_emocoes.most_common():
                porcentagem = (count / len(emocoes_detectadas)) * 100
                print(f"  {emocao}: {count} vezes ({porcentagem:.1f}%)")
        else:
            emocao_predominante = emocao_atual
        
        return emocao_predominante, confianca_atual

def main():
    parser = argparse.ArgumentParser(description="Detector de Emoções Otimizado")
    parser.add_argument("input_video", help="Vídeo de entrada")
    parser.add_argument("-o", "--output", help="Vídeo de saída", default=None)
    parser.add_argument("-m", "--model", help="Modelo LSTM", 
                       default="classificado_emocaoLS.keras")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_video):
        print(f"Erro: Vídeo não encontrado: {args.input_video}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Erro: Modelo não encontrado: {args.model}")
        sys.exit(1)
    
    if args.output is None:
        input_path = Path(args.input_video)
        output_path = input_path.parent / f"{input_path.stem}_emotions{input_path.suffix}"
    else:
        output_path = args.output
    
    try:       
        detector = EmotionVideoDetector(args.model)
        emocao_predominante, confianca_final = detector.processar_video(args.input_video, str(output_path))
        
        print(f"Emoção predominante: {emocao_predominante}")
        print(f"Confiança final: {confianca_final:.1%}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
