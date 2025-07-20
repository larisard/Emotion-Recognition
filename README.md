<h1 align="center"> Classificador de Emoções </h1>

![Badge em Desenvolvimento](http://img.shields.io/static/v1?label=STATUS&message=EM%20DESENVOLVIMENTO&color=GREEN&style=for-the-badge)
![Badge Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)


Sistema de Reconhecimento de emoções através da análise de imagens sequenciais.

- Ferramentas: Linguagem Python, Bibliotecas TensorFlow, Numpy, Keras e OpenCV.

# EmotionRNN

## Arquivos

* `sistema_classificador.py`: Arquivo responsável por criar e treinar o modelo classificador de emoções.
* `aplicacao.py`: Aplicação desenvolvida para utilizar o classificador LSTM e realizar a detecção de emoções em tempo real.
* `haarcascade_frontalface_default.xml`: Arquivo de modelo pré-treinado do Haar Cascade, utilizado para a detecção de faces.
* `classificador_emocaoLS.keras`: O arquivo do modelo classificador de emoções treinado e pronto para uso.

## Como usar a aplicação:

No terminal digite  `python aplicacao.py nome_video.mp4`, onde 
nome_video.mp4 deve ser trocado pelo nome do vídeo a ser analisado

## Pastas

* `S4 Stimuli (Video-clips)`: Contém a base de dados utilizada para o treinamento e validação do modelo, proveniente do Kdef-dyn.
* `Resultados - Treino`: Armazena os resultados obtidos durante a fase de treinamento com o conjunto de teste.
* `Resultados - Aplicação`: Guarda os resultados gerados pela execução da aplicação final.
  
# monografia_Larissa_Sardinha: 

Arquivo PDF da monografia desenvolvida.

  
:construction: Projeto em construção :construction:
