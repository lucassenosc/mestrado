import os
import glob
import pandas as pd

def agregar_relatorios():
    pasta_origem = "relatorios final"
    pasta_destino = "relatorios agregados"
    
    # Verifica se a pasta de origem existe
    if not os.path.exists(pasta_origem):
        print(f"❌ Erro: A pasta '{pasta_origem}' não foi encontrada!")
        return

    # Cria a pasta de destino se não existir
    os.makedirs(pasta_destino, exist_ok=True)
    
    # Pega todos os ficheiros CSV na pasta
    ficheiros_csv = glob.glob(os.path.join(pasta_origem, "*.csv"))
    
    if not ficheiros_csv:
        print(f"⚠️ Nenhum ficheiro CSV encontrado na pasta '{pasta_origem}'.")
        return

    print(f"Encontrados {len(ficheiros_csv)} ficheiros. A iniciar agregação...\n")
    
    # Dicionário para guardar os DataFrames agrupados
    # A chave será a tupla: (Algoritmo, Dataset, Modo)
    agrupamentos = {}
    
    for ficheiro in ficheiros_csv:
        # Extrai o nome do ficheiro sem o caminho e sem a extensão
        nome_ficheiro = os.path.basename(ficheiro).replace(".csv", "")
        
        # A nossa convenção de nomes é: algoritmo_dataset_modo_binarizacao
        # Exemplo 1: standart_wisard_UNSW-NB15_binary_distributive
        # Exemplo 2: decision_tree_Bot-IoT_multiclass
        
        partes = nome_ficheiro.split('_')
        
        # Lógica para descobrir as partes baseada na estrutura do nome
        if len(partes) >= 4:
            if partes[0] == "decision" and partes[1] == "tree":
                algoritmo = "decision_tree"
                dataset = partes[2]
                modo = partes[3]
                binarizacao = "N/A" # Árvores não têm termómetro
            elif partes[0] == "random" and partes[1] == "forest":
                algoritmo = "random_forest"
                dataset = partes[2]
                modo = partes[3]
                binarizacao = "N/A"
            else:
                # É um modelo WiSARD (standart_wisard ou bloom_wisard)
                algoritmo = f"{partes[0]}_{partes[1]}"
                dataset = partes[2]
                modo = partes[3]
                binarizacao = partes[4] if len(partes) > 4 else "N/A"
        else:
            print(f"⚠️ Aviso: Ficheiro com nome fora do padrão ignorado: {nome_ficheiro}")
            continue
            
        chave_grupo = (algoritmo, dataset, modo)
        
        # Lê o CSV atual
        try:
            df_temp = pd.read_csv(ficheiro)
            # Adiciona a coluna da binarização para sabermos a origem dos dados
            if "Binarization" not in df_temp.columns:
                # Inserir no início do dataframe
                df_temp.insert(2, "Binarization", binarizacao) 
            
            # Guarda na nossa lista de agrupamento
            if chave_grupo not in agrupamentos:
                agrupamentos[chave_grupo] = []
            agrupamentos[chave_grupo].append(df_temp)
            
        except Exception as e:
            print(f"❌ Erro ao ler {ficheiro}: {e}")

    # Agora vamos unir e guardar os ficheiros agrupados
    for (algoritmo, dataset, modo), lista_dfs in agrupamentos.items():
        # Concatena todos os dfs desse grupo
        df_final = pd.concat(lista_dfs, ignore_index=True)
        
        # Nome do ficheiro agregado final
        nome_saida = f"{algoritmo}_{dataset}_{modo}_COMPLETO.csv"
        caminho_saida = os.path.join(pasta_destino, nome_saida)
        
       # Guarda no disco
        # sep=';' separa as colunas corretamente para o Excel em PT
        # decimal=',' troca os pontos por vírgulas nos números
        df_final.to_csv(caminho_saida, index=False, sep=';', decimal=',')
        
        print(f"✅ Agregado com sucesso: {nome_saida} (Total de linhas: {len(df_final)})")

    print("\n🚀 PROCESSO DE AGREGAÇÃO CONCLUÍDO!")

if __name__ == "__main__":
    agregar_relatorios()