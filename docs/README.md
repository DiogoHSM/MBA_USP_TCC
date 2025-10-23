# Guia de Análises e Documentação

## Resumo executivo
- **Objetivo**: classificar objetos astronômicos do Sloan Digital Sky Survey (SDSS) nas classes _STAR_, _GALAXY_ e _QSO_ a partir de atributos fotométricos e redshift.
- **Abrangência dos experimentos**: os notebooks implementam desde o preparo dos dados até comparações de múltiplos modelos supervisionados (árvores de decisão, regressão logística, SVM, Naive Bayes, k-NN e Gradient Boosting).
- **Principal insight**: a combinação de normalização `MinMaxScaler` e modelos baseados em árvores (especialmente `HistGradientBoostingClassifier`) mostrou-se consistente para capturar relações não lineares entre bandas fotométricas e resultados de classificação. Os notebooks posteriores investigam alternativas para equilibrar desempenho (acurácia/F1) e custo computacional.
- **Próximos passos sugeridos**: avaliar ajustes de hiperparâmetros para SVM e Gradient Boosting, explorar redução de dimensionalidade (PCA, UMAP) e registrar explicitamente os modelos finalistas em `models/` para reuso.

## Visão geral dos notebooks

| Notebook | Propósito principal | Principais parâmetros e configurações | Métricas registradas |
| --- | --- | --- | --- |
| `01_features_filter.ipynb` | Seleção de atributos relevantes e validação rápida de desempenho com Gradient Boosting. | `train_test_split(test_size=0.2, random_state=1, stratify=y)`; `MinMaxScaler` para normalização; `HistGradientBoostingClassifier(random_state=1)` com atributos `['redshift', 'i', 'u', 'g', 'r', 'z']`. | Acurácia, F1-Score (ponderado), ROC-AUC multiclasse. |
| `02_ml_models_sdss.ipynb` | Treinamento comparativo de modelos clássicos. | `train_test_split(test_size=0.2, random_state=1)`; `DecisionTreeClassifier(random_state=49)`; `LogisticRegression()` (parâmetros padrão); `SVC(C=1.0, kernel='rbf', gamma='scale', random_state=1)`; `GaussianNB()`; `KNeighborsClassifier(n_neighbors=3)`. | Acurácia (teste), relatórios de classificação, matrizes de confusão, escores de treino vs. teste onde aplicável. |
| `03_sdss_modelos_comparacao.ipynb` | Pipeline consolidado para treinar, validar e comparar modelos (inclusive otimizações). | Função `avaliar_modelo` com `cv_folds=5` e `n_jobs=-1` para paralelização; cálculo opcional de ROC-AUC quando `predict_proba` está disponível; geração de matrizes de confusão percentuais. | Acurácia, F1-Score, ROC-AUC, tempo de treino/predição, média e desvio padrão de validação cruzada, matrizes de confusão (absolutas e percentuais). |
| `04_sdss_modelos_comparacao_variante.ipynb` | Espaço para variantes de comparação (ex.: buscas de hiperparâmetros ou experimentos adicionais). | Estrutura espelhada ao notebook `03` para incorporar `GridSearchCV`/`RandomizedSearchCV` e ajustes específicos. | Definir conforme o experimento (recomenda-se manter o conjunto Acurácia/F1/ROC-AUC/tempo). |

## Convenções de nomenclatura e boas práticas

- **Idiomas**: textos explicativos podem permanecer em português; nomes de variáveis, funções e classes devem seguir inglês consistente (`train_data`, `evaluate_model`).
- **Estilo de código**: utilizar `snake_case` para variáveis e funções, `PascalCase` apenas para classes. Evitar abreviações obscuras (`accuracy_score` em vez de `acc_scr`).
- **Notebooks**: manter prefixos numéricos de dois dígitos para refletir a ordem de execução (`01_`, `02_`, ...). Títulos e subtítulos devem usar Markdown descritivo com `#`/`##`.
- **Diretórios e arquivos**: salvar artefatos intermediários em `data/processed/`; modelos serializados em `models/`; figuras em `reports/figures/<experimento>/`. Utilizar nomes em inglês com hífens ou underscore (`roc-curve.png`, `feature_importance.csv`).
- **Boas práticas gerais**: documentar decisões de modelagem em células Markdown, manter seeds (`random_state`) fixos para reprodutibilidade e registrar dependências adicionais no `requirements.txt`.

## Como executar as análises

1. **Dependências mínimas**: Python 3.10+ e pacotes `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`. Instale com `pip install -r requirements.txt`.
2. **Preparação dos dados**: coloque o CSV bruto `sdss.csv` (renomeado a partir do download Kaggle) em `data/raw/`. Alguns notebooks esperam o arquivo no diretório raiz; se necessário, ajuste o caminho ou copie para `notebooks/`.
3. **Ordem recomendada**:
   1. `01_features_filter.ipynb` – confirmar limpeza, seleção de atributos e baseline com Gradient Boosting (≈5–10 minutos em CPU padrão).
   2. `02_ml_models_sdss.ipynb` – comparar algoritmos clássicos e registrar métricas (≈10–15 minutos dependendo do tamanho do dataset).
   3. `03_sdss_modelos_comparacao.ipynb` – consolidar resultados, validar com _k_-fold (≈15–25 minutos; geração de gráficos pode aumentar o tempo).
   4. `04_sdss_modelos_comparacao_variante.ipynb` – executar apenas quando for necessário testar novas configurações (tempo variável conforme busca de hiperparâmetros).
4. **Boas práticas de execução**: antes de rodar notebooks de comparação, garanta que `data/processed/` contenha subconjuntos limpos para reutilização. Utilize _checkpoints_ (`File > Save and Checkpoint`) e exporte gráficos para `reports/figures/`. Anote tempos reais de execução em Markdown ao final de cada notebook para manter o histórico atualizado.

## Referências rápidas

- **Resultados consolidados**: priorize registrar tabelas finais no `03_sdss_modelos_comparacao.ipynb` e exportá-las para `reports/` (CSV ou imagens) para fácil consulta.
- **Contato**: dúvidas sobre organização ou estilo podem ser anotadas em _issues_ do repositório ou adicionadas a futuras revisões deste documento.
