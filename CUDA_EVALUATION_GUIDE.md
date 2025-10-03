# Guia de Avaliação CUDA com ParvEval

Este guia mostra como avaliar uma LLM refinada para programação CUDA usando o benchmark ParvEval.

## Visão Geral

O ParvEval contém **60 problemas CUDA** que cobrem:

- **Geometria**: closest pair, convex hull, triangulação
- **Transformações**: ReLU, mapeamentos, operações elemento-wise
- **Reduções**: soma, produto, XOR, médias
- **Álgebra Linear**: multiplicação matriz-vetor, decomposições
- **Stencils**: operações de vizinhança
- **Sorting**: algoritmos de ordenação paralela

## Pré-requisitos

### Software Necessário

- Python ≥3.7
- Compilador C++ com suporte a C++17 e OpenMP
- Make e CMake
- NVIDIA GPU com CUDA toolkit instalado
- Sua LLM refinada para CUDA

### Instalação

1. **Clone o repositório**:

```bash
git clone --recurse-submodules https://github.com/parallelcodefoundry/ParEval.git
cd ParEval
```

2. **Instale dependências Python**:

```bash
pip install -r requirements.txt
```

3. **Compile os drivers C++**:

```bash
cd drivers/cpp
make
cd ../..
```

## Passo a Passo para Avaliação CUDA

### Passo 1: Extrair Prompts CUDA

Execute o script para criar um arquivo apenas com prompts CUDA:

```bash
python3 create_cuda_prompts.py
```

Isso criará `prompts/cuda-only-prompts.json` com 60 problemas CUDA específicos.

### Passo 2: Gerar Código com sua LLM

```bash
python3 generate/generate.py \
    --prompts prompts/cuda-only-prompts.json \
    --model deepseek-ai/deepseek-coder-1.3b-base \
    --output result-deepseek-coder-1.3b-base.json \
    --num_samples_per_prompt 50 \
    --temperature 0.2 \
    --prompted \
    --max_new_tokens 1024 \
    --restart
```

**Parâmetros importantes**:

- `--model`: Caminho para sua LLM ou handle do HuggingFace
- `--num_samples_per_prompt`: Número de soluções por problema (padrão: 50)
- `--prompted`: Adiciona comentários de solução (recomendado)
- `--temperature`: Controla aleatoriedade (0.2 é bom para código)

### Passo 3: Avaliar o Código Gerado

```bash
python3 drivers/run-all.py cuda-results.json \
    --include-models cuda \
    --output cuda-evaluation.json \
    --yes-to-all \
    --build-timeout 60 \
    --run-timeout 30
```

**Parâmetros importantes**:

- `--include-models cuda`: Testa apenas CUDA
- `--yes-to-all`: Responde sim para todas as confirmações
- `--build-timeout`: Timeout para compilação (segundos)
- `--run-timeout`: Timeout para execução (segundos)

### Passo 4: Calcular Métricas

```bash
# Converter resultados para CSV
python3 analysis/create-dataframe.py cuda-evaluation.json --output cuda-results.csv

# Calcular métricas finais (k = 1,5,10,20 por padrão)
python3 analysis/metrics.py cuda-results.csv --output cuda-metrics.csv

# Ou especificar valores de k customizados
python3 analysis/metrics.py cuda-results.csv --k 1 10 50 100 --output cuda-metrics.csv
```

### Passo 5: Analisar Resultados

As métricas calculadas incluem:

- **pass@k**: Taxa de sucesso (compilação + execução + correção)
- **build@k**: Taxa de compilação bem-sucedida
- **efficiency@k**: Taxa de correção dos resultados
- **speedup@k**: Ganho de performance vs implementação serial

Visualize os resultados:

```bash
# Ver métricas resumidas
cat cuda-metrics.csv

# Ver resultados detalhados por problema
head -20 cuda-results.csv
```

## Comandos Rápidos

Para uma avaliação rápida com menos amostras:

```bash
# Gerar (10 amostras por problema)
python3 generate/generate.py \
    --prompts prompts/cuda-only-prompts.json \
    --model sua-llm \
    --output quick-cuda.json \
    --num_samples_per_prompt 10 \
    --prompted

# Avaliar
python3 drivers/run-all.py quick-cuda.json \
    --include-models cuda \
    --output quick-evaluation.json \
    --yes-to-all

# Métricas
python3 analysis/create-dataframe.py quick-evaluation.json --output quick-results.csv
python3 analysis/metrics.py quick-results.csv --output quick-metrics.csv
```

## Configurações Avançadas

### Para APIs (OpenAI, Gemini)

Se sua LLM está disponível via API:

```bash
# OpenAI
python3 generate/generate-openai.py \
    --prompts prompts/cuda-only-prompts.json \
    --model gpt-4 \
    --output cuda-openai.json

# Gemini
python3 generate/generate-gemini.py \
    --prompts prompts/cuda-only-prompts.json \
    --model gemini-pro \
    --output cuda-gemini.json
```

### Configuração de Ambiente

Para sistemas com múltiplas GPUs ou configurações específicas:

```bash
# Especificar GPU
export CUDA_VISIBLE_DEVICES=0

# Configurar diretório scratch (se /tmp não funcionar)
python3 drivers/run-all.py cuda-results.json \
    --include-models cuda \
    --scratch-dir /path/to/shared/scratch \
    --output cuda-evaluation.json
```

## Interpretação dos Resultados

### Métricas Principais

O sistema calcula automaticamente para **k = [1, 5, 10, 20]** (personalizável):

- **pass@k**: Probabilidade de ≥1 solução correta entre as k melhores
- **build@k**: Probabilidade de ≥1 compilação bem-sucedida entre as k melhores
- **speedup@k**: Speedup esperado escolhendo a melhor entre k amostras
- **efficiency@k**: Eficiência esperada (speedup/recursos) entre k amostras

**Exemplo**: pass@10 = 0.85 significa 85% de chance de ter pelo menos 1 solução correta entre as 10 melhores tentativas.

### Comparação com Leaderboard

Compare seus resultados com o [ParvEval Leaderboard](https://pssg.cs.umd.edu/blog/2024/pareval/) oficial.

## Troubleshooting

### Problemas Comuns

1. **Erro de compilação CUDA**:

   - Verifique se CUDA toolkit está instalado
   - Confirme que `nvcc` está no PATH

2. **Timeout durante execução**:

   - Aumente `--run-timeout` para problemas complexos
   - Verifique se GPU tem memória suficiente

3. **Erro de memória**:

   - Reduza `--batch_size` na geração
   - Use `--num_samples_per_prompt` menor

4. **Problemas com /tmp**:
   - Use `--scratch-dir` apontando para diretório compartilhado

## Estrutura de Arquivos Gerados

```
cuda-results.json          # Código gerado pela LLM
cuda-evaluation.json       # Resultados da avaliação
cuda-results.csv          # Dados tabulares detalhados
cuda-metrics.csv          # Métricas finais resumidas
```

## Próximos Passos

Após a avaliação, você pode:

1. **Analisar falhas**: Identificar padrões nos erros de compilação/execução
2. **Refinar modelo**: Usar resultados para melhorar treinamento
3. **Comparar modelos**: Avaliar diferentes versões da sua LLM
4. **Contribuir**: Submeter resultados para o leaderboard oficial

## Recursos Adicionais

- [Paper ParvEval](https://pssg.cs.umd.edu/assets/papers/2024-06-pareval-hpdc.pdf)
- [Leaderboard Oficial](https://pssg.cs.umd.edu/blog/2024/pareval/)
- [Repositório GitHub](https://github.com/parallelcodefoundry/ParEval)
