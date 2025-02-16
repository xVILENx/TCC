# TCC - Projeto de Extração de Informações Biomédicas em Abstracts Científicos

Este repositório apresenta um **projeto completo de extração de informações biomédicas**, com foco em **genes** e **doenças** específicos (COVID-19 e Alzheimer), utilizando **Modelos de NLP (BioBERT, SciBERT, PubMedBERT)**. O objetivo principal é identificar automaticamente, em abstracts científicos, as menções a genes e doenças relevantes para a pesquisa biomédica.

---

## 1. Visão Geral

- **Tema**: Extração de Informações Biomédicas (genes e doenças) em abstracts científicos.  
- **Ferramentas**:  
  - Linguagem Python (≥ 3.7).  
  - Bibliotecas do Hugging Face, como `transformers` e `datasets`.  
  - Modelos pré-treinados: BioBERT, SciBERT e PubMedBERT.  
- **Objetivo**:  
  - Construir um pipeline para **ler e pré-processar** abstracts.  
  - Aplicar modelos de NLP para **classificar** e **extrair** menções de genes e doenças.  
  - Comparar desempenho **antes e depois do fine-tuning** usando uma **lista de genes** curada.  
  - Realizar testes de avaliação (Accuracy, Precision, Recall, F1-Score) e testes adicionais em **textos não vistos**.

Este projeto é útil em cenários onde se necessita **minerar** textos científicos para identificar correlações entre genes e doenças de grande relevância, como COVID-19 ou Alzheimer’s Disease, e **facilitar** o trabalho de revisão e análise em pesquisas biomédicas.

---

## 2. Estrutura do Repositório
- **genes_list.txt**: Arquivo texto com genes relevantes (um por linha).  
- **abstracts.json**: Arquivo JSON (ou outro formato) contendo abstracts coletados do PubMed, ou subdividido em Alzheimer vs. COVID-19.  
- **scripts/no_finetuning_demo.py**: Exemplo de código que apenas carrega modelos pré-treinados sem ajustá-los.  
- **scripts/finetuning_with_genes.py**: Versão principal que **lê a lista de genes**, **treina** (fine-tuning) e **avalia** o modelo.

---

## 3. Dependências e Ambiente

### 3.1 Instalação de Dependências

Para garantir funcionamento adequado, criar e ativar um ambiente virtual (opcional) e instalar as bibliotecas listadas em `requirements.txt`:
transformers==4.28.0
torch==2.0.0
scikit-learn==1.1.3
matplotlib==3.6.0
seqeval==1.2.2
```bash
# Em um ambiente Linux ou MacOS, por exemplo:
python3 -m venv venv
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
