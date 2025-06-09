# Sistema Inteligente para Análise de Apendicite Pediátrica

Este projeto utiliza machine learning (Random Forest) para apoiar profissionais de saúde no diagnóstico, decisão de manejo e avaliação de severidade de apendicite em crianças, com base em dados reais. O sistema realiza pré-processamento, treinamento de modelos e demonstração de predição para novos pacientes.

## Principais Funcionalidades

- Download automático da base de dados pública de apendicite pediátrica.
- Pré-processamento dos dados, incluindo tratamento de valores ausentes e codificação de variáveis.
- Treinamento de modelos Random Forest para:
  - Previsão do tipo de manejo (conservador ou cirúrgico).
  - Previsão da severidade (simples ou complexa) para casos cirúrgicos.
- Avaliação dos modelos utilizando validação cruzada.
- Demonstração de uso do sistema com um paciente exemplo.

## Como Executar

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

2. Execute o script principal:
   ```
   python appendicitis_analysis.py
   ```

O script irá baixar automaticamente a base de dados, treinar os modelos e mostrar exemplos de uso.

## Estrutura do Projeto

- `appendicitis_analysis.py`: script principal de análise de apendicite.
- `requirements.txt`: dependências do projeto.
- `dev-requirements.txt`: dependências adicionais para desenvolvimento e testes.
- Pasta `tests/`: contém testes automatizados para scripts de exemplo de análise de organizações.
- Scripts auxiliares (`calculations.py`, `revenue_visual.py`): exemplos de análise de dados de organizações, não relacionados à análise de apendicite.

## Observações

- Os scripts de organizações (`calculations.py`, `revenue_visual.py` e arquivos em `tests/`) utilizam um arquivo `data.csv` que não é necessário para a análise de apendicite.
- O foco principal do projeto é o script `appendicitis_analysis.py`.

## Requisitos

- Python 3.x
- pandas, scikit-learn, requests, matplotlib, pytest

---

Projeto adaptado para fins acadêmicos e de demonstração.