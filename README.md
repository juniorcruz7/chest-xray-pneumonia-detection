# 🩺 Chest X-Ray Pneumonia Detection

---

# 📌 Objetivo

Desenvolver um modelo robusto capaz de classificar imagens de raio-X de tórax em:

- 0 → Normal  
- 1 → Pneumonia  

O projeto foi estruturado com foco em:

- Reprodutibilidade
- Tratamento de desbalanceamento
- Avaliação com métricas adequadas para saúde
- Organização clara do pipeline
- Separação entre treino, validação, teste e inferência

---

# 🧠 Arquitetura do Modelo

Backbone utilizado:

DenseNet121 pré-treinada no ImageNet.

A camada final foi substituída por uma camada linear com saída única para classificação binária:

```
model.classifier = nn.Linear(num_features, 1)
```

Função de perda:

BCEWithLogitsLoss com ajuste de `pos_weight` para compensar desbalanceamento de classes.

Otimizador:

Adam (learning rate = 1e-4)

---

# 🔄 Pipeline do Projeto

## 1️⃣ Configuração Global

- Fixação de seed (42)
- Determinismo ativado no cuDNN
- Detecção automática de CPU/GPU

Isso garante reprodutibilidade dos experimentos.

---

## 2️⃣ Split Estratificado

Divisão em duas etapas:

1. Treino / Teste Interno (80/20)
2. Treino / Validação (80/20 dentro do treino)

O uso de `stratify` mantém a proporção entre classes.

---

## 3️⃣ Data Augmentation (Treino)

- Resize 224x224
- Random Horizontal Flip
- Random Rotation
- Random Affine
- Color Jitter
- Normalização (padrão ImageNet)

Objetivo: reduzir overfitting e melhorar generalização.

---

## 4️⃣ Tratamento de Desbalanceamento

Foi aplicado peso positivo na função de perda:

```
pos_weight = num_normal / num_pneumonia
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

Isso penaliza mais erros na classe minoritária.

---

## 5️⃣ Treinamento

- Batch size: 32
- Epochs: 5
- Métrica principal: ROC-AUC
- Salvamento automático do melhor modelo baseado na maior ROC-AUC de validação

O melhor modelo é salvo como:

```
best_model.pth
```

---

## 6️⃣ Avaliação

Avaliação no teste interno inclui:

- ROC-AUC
- Matriz de Confusão
- Precision
- Recall
- F1-Score
- Classification Report

Threshold padrão utilizado: 0.5

---

## 7️⃣ Inferência e Submissão

O modelo salvo é carregado e utilizado para gerar probabilidades no conjunto de teste externo.

A saída é gerada no formato:

```
id,target
img_0001.jpeg,0.87
img_0002.jpeg,0.02
...
```

Arquivo gerado:

```
submission.csv
```

---

# 📁 Estrutura do Projeto

```
.
├── data/
│   ├── train/
│   ├── test_images/
│   └── test.csv
│
├── pneumonia_detection.ipynb
├── best_model.pth
├── submission.csv
├── requirements.txt
└── README.md
```

---

# 🚀 Como Usar

## 🔹 1. Clonar o repositório

```
git clone https://github.com/juniorcruz7/chest-xray-pneumonia-detection.git
cd chest-xray-pneumonia-detection
```

---

## 🔹 2. Criar ambiente virtual (opcional, recomendado)

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```
python3 -m venv venv
source venv/bin/activate
```

---

## 🔹 3. Instalar dependências

```
pip install -r requirements.txt
```

---

## 🔹 4. Organizar os dados

A pasta `data/` deve conter:

```
data/
 ├── train/
 │    ├── NORMAL/
 │    └── PNEUMONIA/
 ├── test_images/
 └── test.csv
```

Os dados devem ser baixados e colocados em suas respectivas páginas através do link exclusivo do desafio: https://www.kaggle.com/competitions/ligia-compviz/data

---

## 🔹 5. Executar o treinamento

Abra:

```
pneumonia_detection.ipynb
```

Execute todas as células.

O modelo será treinado e o melhor peso será salvo automaticamente.

---

## 🔹 6. Gerar submissão

Ao final do notebook, a etapa de inferência irá gerar:

```
submission.csv
```

Pronto para envio no Kaggle.

---

# 📊 Métrica Principal

ROC-AUC

Justificativa:
- Mais robusta para dados desbalanceados
- Avalia capacidade discriminativa independentemente do threshold

---

# 💻 Compatibilidade

- CPU
- GPU (CUDA)

O dispositivo é selecionado automaticamente:

```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

# 🔬 Melhorias Futuras

- Fine-tuning completo do backbone
- Early Stopping
- Learning Rate Scheduler
- Cross-validation (K-Fold)
- Grad-CAM para interpretabilidade
- Test-Time Augmentation
- Mixed Precision Training
- Deploy via API (FastAPI)

---

# ⚖️ Considerações Éticas

Este modelo:

- Não possui validação clínica
- Não deve ser utilizado como ferramenta diagnóstica isolada
- É destinado exclusivamente para fins educacionais e experimentais

Aplicações clínicas exigem:
- Validação externa
- Aprovação regulatória
- Avaliação de vieses populacionais
