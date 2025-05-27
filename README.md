# Descrição do problema

  O algoritmo tem como objetivo classificar imagens de dois tipos de animais: gatos e cachorros. Para isso, ele utiliza um modelo de rede neural convolucional (CNN) treinado com imagens do dataset CIFAR-10 passado como exemplo em sala de aula. É um problema de classificação de imagens, que o algoritmo consegue distinguir entre gatos (0) e cachorros (1)

# Justificativa das técnicas utilizadas

classes_to_use = [3, 5]  # gato=3, cachorro=5, como só queremos classificar gatos e cachorros, foi pego o índice de cada um para as classificar as respectivas classes.

Foi dividido em conjustos de testes e treinamento como solicitado, utilizando a postura de 80% para treino e 20% para teste, utilizando também o random_state = 42 para garantir que o output vai ser sempre o mesmo.

##**layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3))**

Na construção da CNN, foi aplicado 32 kernels de tamanho 3x3 sobre a imagem e depois usado a função de ativação 'RELU', a função de ativação RELU ajuda a melhorar o treinamento e evitar o problema do gradiente zero.


##**layers.MaxPooling2D((2,2)),**

Redimensionamento da imagem para um valor máximo de 2x2.

##**layers.Conv2D(64, (3,3), activation='relu'),**

Agora foi aplicado 64 kernels de tamanho 3x3 sobre a imagem e também foi utilzado a função de ativação 'RELU'

##**layers.Flatten(),**

Transforma as características em um vetor de 1 dimensão

**layers.Dense(64, activation='relu'),**

Uma camada com 64 neurônios de saída e usando ativação 'RELU'.

**layers.Dense(1, activation='sigmoid'),**

Uma camada de saída com 1 neurônio, a função sigmoid retorna um valor entre 0 e 1, muito boa pois estamos utilizando classificação binária.

**model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])**

Configuração do modelo para treinar, optimizer='adam' algoritmo de otimização que ajusta os pesos, loss='binary_crossentropy' é a função de perda, mede a diferença entre a predição.

**Cálculo das métricas**

Foi feito as métricas de precisão, recall e f1-score, a f1-score é a mais importante pois mostra o balanceamentro entre a precisão e o recall,

> **Etapas realizadas**

- Importação das bibliotecas
- Carregamento e normalização do dataset
- Filtragem para usar apenas as classes de gatos e cachorros
- Mapeamento de rótulos entre 0(gatos) e 1(cachorros)
- Divisão de dados em treinamento e teste
- Criação da Arquitetura da CNN
- Compilação do modelo
- Treinamento do modelo
- Predição e métricas avaliadas
- Carregamento das imagens originais para serem classificadas pelo modelo de classificação.

> **Resultados Obtidos**

Precisão: 0.8065
Recall: 0.6808
F1-Score: 0.7384

O modelo previu 5 imagens corretas das 6 imagens, ele não conseguiu classificar corretamente 1 imagem.

> **Tempo Total Gasto**

O treinamento demorou 200 segundos para finalizar.

> **Dificuldades encontradas**

Uma das dificuldades que eu encontrei foi o overfitting, conforme eu tentava arrumar os parâmetros, a accuracy acabava piorando.
