# Mestrado

Repositório com códigos, notebooks, scripts auxiliares e relatórios relacionados aos experimentos desenvolvidos no projeto de mestrado.

O projeto contém experimentos com modelos tradicionais de aprendizado de máquina e modelos WiSARD, incluindo versões Standard WiSARD e Bloom WiSARD. Também há scripts para agregação de relatórios e arquivos auxiliares usados na execução dos experimentos.

## Estrutura do repositório

```text
.
├── libs/
├── relatorios agregados/
├── relatorios final/
├── agregador.py
├── agregador_csv.ipynb
├── bloom_wisard_final.ipynb
├── bloom_worker_runtime.py
├── build_data.ipynb
├── build_data3.ipynb
├── curva_roc.ipynb
├── decision_tree.ipynb
├── random_forest.ipynb
├── standard_worker_runtime.py
├── standart_wisard.ipynb
├── svm.ipynb
├── run_all.bat
├── requirements.txt
├── .gitignore
└── .gitattributes
```

## Arquivos principais

* `build_data.ipynb`: notebook usado para preparação/construção dos dados.
* `build_data3.ipynb`: notebook usado para preparação/construção de uma segunda etapa ou variação dos dados.
* `decision_tree.ipynb`: experimentos com árvore de decisão.
* `random_forest.ipynb`: experimentos com random forest.
* `svm.ipynb`: experimentos com SVM.
* `standart_wisard.ipynb`: experimentos com Standard WiSARD.
* `bloom_wisard_final.ipynb`: experimentos com Bloom WiSARD.
* `curva_roc.ipynb`: geração/análise de curvas ROC.
* `agregador.py`: script para agregar relatórios CSV gerados pelos experimentos.
* `agregador_csv.ipynb`: notebook para agregação/análise de arquivos CSV.
* `bloom_worker_runtime.py`: módulo auxiliar usado na execução paralela da Bloom WiSARD.
* `standard_worker_runtime.py`: módulo auxiliar usado na execução paralela da Standard WiSARD.
* `run_all.bat`: script de execução automática no Windows.
* `requirements.txt`: lista de dependências Python necessárias para reproduzir o ambiente.

## Dados

Os dados grandes não estão versionados neste repositório, pois excedem o tamanho recomendado para repositórios Git/GitHub.

As pastas de dados devem ser baixadas separadamente pelo Google Drive:

[Baixar dados no Google Drive](https://drive.google.com/drive/folders/1xaqv2979bKpBT411Ifh70MKPlZ1Dz0id?usp=drive_link)

Após o download, mantenha a seguinte estrutura na raiz do projeto:

```text
data/
data2/
data3/
```

Essas pastas estão no `.gitignore` e não devem ser enviadas ao GitHub.

## Requisitos

Este projeto utiliza Python e as dependências listadas em `requirements.txt`.

Recomenda-se criar um ambiente virtual antes de instalar as dependências.

## Criação do ambiente virtual no Windows

No Git Bash, dentro da pasta do projeto:

```bash
py -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Relatórios

Os resultados gerados pelos experimentos são armazenados em:

```text
relatorios final/
relatorios agregados/
```

O script `agregador.py` lê os relatórios individuais em `relatorios final/` e gera arquivos agregados em `relatorios agregados/`.

Para executar o agregador:

```bash
python agregador.py
```

## Reprodutibilidade

Para reproduzir os experimentos em outro computador:

1. Clone o repositório.
2. Crie o ambiente virtual.
3. Instale as dependências com `requirements.txt`.
4. Baixe as pastas de dados pelo Google Drive.
5. Coloque `data/`, `data2/` e `data3/` na raiz do projeto.
