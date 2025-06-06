import numpy as np
import cv2
from tqdm import tqdm # Adicionando a importação que faltava

# ==================== Extrator HOG (Versão do Treinamento v5.py) ====================
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
        """Calcula o vetor de características HOG para uma lista de imagens."""
        hog_vetores = []
        print(f"Calculando HOG customizado para {len(imagens)} frames...") # Mantendo o print original
        # Adicionando o loop com tqdm que causou o NameError original
        for img in tqdm(imagens, desc="Custom HOG Features"):
            # Garante que a imagem tenha o tamanho correto e seja float32
            if img.shape != tamanho_imagem:
                img = cv2.resize(img, tamanho_imagem)
            img = img.astype(np.float32)

            magnitude, angulo = HOGFeatureExtractor.calcularGradiente(img)

            altura, largura = img.shape
            celula_x = largura // tamanho_celula
            celula_y = altura // tamanho_celula

            # Verifica se os tamanhos são divisíveis (mantendo a lógica original de v5.py)
            if largura % tamanho_celula != 0 or altura % tamanho_celula != 0:
                 print(f"Erro HOG: Tamanho da imagem ({largura},{altura}) não é divisível pelo tamanho da célula ({tamanho_celula}). Pulando imagem.")
                 est_blocos_y = celula_y - tamanho_bloco + 1
                 est_blocos_x = celula_x - tamanho_bloco + 1
                 tamanho_esperado = est_blocos_y * est_blocos_x * tamanho_bloco * tamanho_bloco * bins
                 # Adiciona vetor de zeros se houver erro, como no v5.py
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
                            # Evita divisão por zero se bins for 1 (lógica de v5.py)
                            bin_idx = int(ang_celula[y, x] / (180.0 / bins)) % bins if bins > 0 else 0
                            hist[bin_idx] += mag_celula[y, x]
                    histograma_orientacoes[i, j] = hist

            blocos_y = celula_y - tamanho_bloco + 1
            blocos_x = celula_x - tamanho_bloco + 1
            blocos_normalizados = []

            for y in range(blocos_y):
                for x in range(blocos_x):
                    bloco = histograma_orientacoes[y:y+tamanho_bloco, x:x+tamanho_bloco].flatten()
                    # Normalização L2-Hys (L2 seguido de clipping e L2 novamente) - como em v5.py
                    norm = np.linalg.norm(bloco) + 1e-6 # Evita divisão por zero
                    bloco_normalizado = bloco / norm
                    bloco_normalizado = np.minimum(bloco_normalizado, 0.2) # Clipping
                    norm = np.linalg.norm(bloco_normalizado) + 1e-6 # Renormaliza
                    bloco_normalizado = bloco_normalizado / norm
                    blocos_normalizados.append(bloco_normalizado)

            # Concatena todos os blocos normalizados para formar o vetor HOG final (como em v5.py)
            vetor_hog = np.concatenate(blocos_normalizados) if blocos_normalizados else np.array([])
            hog_vetores.append(vetor_hog)

        # Verifica se todos os vetores têm o mesmo tamanho (como em v5.py)
        if hog_vetores:
            # Calcula tamanho_ref com base no primeiro vetor válido
            tamanho_ref = 0
            primeiro_valido_idx = -1
            for idx, v in enumerate(hog_vetores):
                if v.size > 0:
                    tamanho_ref = len(v)
                    primeiro_valido_idx = idx
                    break
            
            if primeiro_valido_idx != -1: # Se encontrou algum vetor válido
                for i, v in enumerate(hog_vetores):
                    if len(v) != tamanho_ref:
                        # Se o vetor atual for inválido (tamanho 0), preenche com zeros
                        if len(v) == 0 and tamanho_ref > 0:
                             print(f"Aviso HOG: Vetor {i} estava vazio. Preenchendo com zeros ({tamanho_ref}).")
                             hog_vetores[i] = np.zeros(tamanho_ref)
                        # Se o vetor atual tem tamanho diferente do ref, ajusta
                        elif len(v) != 0:
                            print(f"Aviso HOG: Vetor {i} tem tamanho {len(v)}, esperado {tamanho_ref}. Ajustando e preenchendo com zeros.")
                            vetor_ajustado = np.zeros(tamanho_ref)
                            len_min = min(len(v), tamanho_ref)
                            vetor_ajustado[:len_min] = v[:len_min]
                            hog_vetores[i] = vetor_ajustado
            else:
                print("Aviso HOG: Todos os vetores HOG calculados estavam vazios.")


        hog_vetores_np = np.array(hog_vetores)
        # Verifica se o array resultante não está vazio antes de imprimir o shape
        if hog_vetores_np.size > 0:
            print(f"Shape features HOG (custom): {hog_vetores_np.shape}") # Mantendo print original
        else:
            print("Aviso HOG: Array final de vetores HOG está vazio.")
            # Retorna um array com shape esperado (0, num_features) se vazio
            # O número de features esperado é blocos_y * blocos_x * tamanho_bloco * tamanho_bloco * bins
            # Recalcula blocos_y/x caso não tenham sido definidos (se todas imgs falharam)
            altura, largura = tamanho_imagem
            celula_x = largura // tamanho_celula
            celula_y = altura // tamanho_celula
            blocos_y_esperado = celula_y - tamanho_bloco + 1
            blocos_x_esperado = celula_x - tamanho_bloco + 1
            num_features_esperado = blocos_y_esperado * blocos_x_esperado * tamanho_bloco * tamanho_bloco * bins
            return np.empty((0, num_features_esperado)) 
            
        return hog_vetores_np # Retorna o array de vetores HOG planos


