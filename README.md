# MBA USP TCC – Classificação de Objetos do SDSS

Este repositório organiza o fluxo de trabalho do TCC para classificação de objetos astronômicos do Sloan Digital Sky Survey (SDSS). O objetivo é facilitar a reprodução dos experimentos, desde a coleta e preparação dos dados até a análise dos resultados.

## Estrutura de pastas

```
├── data
│   ├── processed/   # artefatos derivados dos notebooks ou scripts de preparação
│   └── raw/         # dados originais obtidos da fonte
├── docs/            # documentação complementar (slides, artigos, notas, etc.)
├── models/          # modelos treinados e artefatos relacionados
├── notebooks/       # cadernos Jupyter numerados conforme a ordem de execução
├── reports/
│   └── figures/     # figuras e gráficos organizados por experimento
└── src/             # scripts reutilizáveis para automações e pipelines
```

## Dados: obtenção e preparo
1. Acesse o conjunto **Sloan Digital Sky Survey (SDSS) DR14** disponível no Kaggle ([lucidlenn/sloan-digital-sky-survey](https://www.kaggle.com/datasets/lucidlenn/sloan-digital-sky-survey)).
2. Baixe o arquivo CSV original (`Skyserver_SQL2_27_2018 6_51_39 PM.csv`) e renomeie para `sdss.csv` para manter a consistência com os notebooks.
3. Salve o arquivo em `data/raw/`.
4. Utilize o notebook `01_features_filter.ipynb` para aplicar filtros, seleção de atributos e gerar versões tratadas do conjunto. Salve saídas intermediárias ou finais em `data/processed/` para reutilização em execuções futuras.

## Fluxo de trabalho analítico
1. **Preparação de dados** – `01_features_filter.ipynb` carrega `data/raw/sdss.csv`, executa limpeza, seleção de atributos e pode exportar subconjuntos tratados para `data/processed/`.
2. **Treinamento de modelos** – `02_ml_models_sdss.ipynb` treina diferentes algoritmos supervisionados utilizando os dados tratados.
3. **Comparação principal de modelos** – `03_sdss_modelos_comparacao.ipynb` consolida métricas de avaliação, tabelas e gráficos finais.
4. **Variação de comparação** – `04_sdss_modelos_comparacao_variante.ipynb` registra experimentos alternativos ou ajustes específicos.

Scripts reutilizáveis devem ser implementados em `src/`, permitindo automatizar etapas recorrentes (ex.: preparação de dados em lote ou avaliação de modelos). Modelos finais serializados (pickle, joblib, ONNX) ficam em `models/`.

> ℹ️ Consulte o guia detalhado em [`docs/README.md`](docs/README.md) para um panorama dos notebooks, convenções de nomenclatura, métricas monitoradas e instruções passo a passo de execução.

## Relatórios e figuras
Os gráficos gerados pelos notebooks devem ser exportados para `reports/figures/`, agrupados por experimento (por exemplo, `reports/figures/base/` e `reports/figures/pre_paralelismo/`). Isso facilita a incorporação em relatórios, artigos ou apresentações.

## Próximos passos sugeridos
- Documentar dependências e ambiente de execução (por exemplo, um arquivo `environment.yml` ou `requirements.txt`).
- Converter notebooks críticos em scripts dentro de `src/` para permitir execuções automatizadas ou agendadas.
- Registrar no diretório `docs/` quaisquer relatórios intermediários, atas ou apresentações produzidas ao longo do projeto.
