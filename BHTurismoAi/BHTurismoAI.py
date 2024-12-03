import os
import pandas as pd
import chardet
import codecs
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from gensim.models import Word2Vec
import gensim



# Baixar recursos do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Diretórios e caminhos utilizados:
rawDataFolder = "./Data/arquivos_csv" 
completeDatabasePath = "./Data/base completa.csv"
reducedDataBasePath = "./Data/base reduzida.csv"
preprocessedDataPath = "./Data/dados_preprocessados.csv"
embeddingsPath = "./embeddings_turisticos.csv"
inputW2VPath = "./modelo_descricao_word2vec_turistico.model"


stop_words = set(stopwords.words('portuguese'))


def readAndUnifyFiles(folderPath, completeDatabasePath):
    with open(completeDatabasePath, 'w', encoding="utf-8") as out_file:
        # Percorre todos os arquivos no diretório
        for filename in os.listdir(folderPath):
            if filename.endswith('.csv'):
                file_path = os.path.join(folderPath, filename)
                
                encoding = detectEncoding(file_path)
                # Lê o conteúdo de cada arquivo e escreve no arquivo de saída
                with open(file_path, 'r', encoding=encoding) as file:
                    content = file.read()
                    out_file.write(content)

def removeAllOccurrences(file_path, sentence):
    encoding = detectEncoding(file_path)
    with open(file_path, 'r+', encoding=encoding) as file:
        content = file.read()

        # Remove all occurrences of the sentence
        new_content = content.replace(sentence, '')

        # Move the file pointer to the beginning and write the new content
        file.seek(0)
        file.write(new_content)
        file.truncate()

def addSentenceAtBeginning(file_path, sentence):
    encoding = detectEncoding(file_path)
    with open(file_path, 'r+', encoding=encoding) as file:
        # Read the existing content of the file
        content = file.read()
        
        # Move the pointer to the beginning of the file
        file.seek(0)
        
        # Write the sentence at the beginning, followed by the original content
        file.write(sentence + content)

def readFirstLine(file_path):
    encoding = detectEncoding(file_path)
    with open(file_path, 'r', encoding=encoding) as file:
        first_line = file.readline()
        return first_line

def removeExcessHeaders(filePath):
    header = readFirstLine(filePath)
    #print("header:\n"+header)
    removeAllOccurrences(filePath, header)
    #print("removi todos os headers")
    addSentenceAtBeginning(filePath, header)
    #print("coloquei o primeiro header novamente!")

def removeLinesNotStartingWithNumber(file_path):
    header = readFirstLine(file_path)
    encoding = detectEncoding(file_path)
    with open(file_path, 'r+', encoding=encoding) as file:
        lines = file.readlines()  # Read all lines into a list
        
        # Filter lines that start with a number
        filtered_lines = [line for line in lines if line.strip() and line[0].isdigit()]
        
        # Move the file pointer to the beginning and overwrite the file
        file.seek(0)
        file.writelines(filtered_lines)
        file.truncate()  # Remove any remaining content that was not overwritten
    addSentenceAtBeginning(file_path, header)
    
def removeIncompleteLines(arquivo_entrada, arquivo_saida, quantidadeCampos):
    # Lê o arquivo CSV, ignorando a primeira linha (cabeçalho)
    encoding = detectEncoding(arquivo_entrada)
    #print("encoding: "+encoding)
    content = ""
    with open(arquivo_entrada, 'r', encoding=encoding) as in_file:
        for linha in in_file:
            campos = linha.split(";")
            if(len(campos) == quantidadeCampos):
                #out_file.write(linha)
                content = content+linha

    with open(arquivo_saida, 'w', encoding="utf-8") as out_file:
        out_file.write(content)

def removeIncompatibleLines(arquivo_entrada, arquivo_saida):
    encoding = detectEncoding(arquivo_entrada)
    #print("encoding: "+encoding)
    content = ""
    with open(arquivo_entrada, 'r', encoding=encoding) as in_file:
        linhas = in_file.readlines()
        content = content + linhas[0]
        for i in range(1, len(linhas)):
            registro = linhas[i]
            campos = registro.split(";")
            condition = campos[0].isdigit() #id precisa ser numerico
            condition = condition and (not campos[1] == False) #o campo de descrição nao pode estar vazio
            if(condition):
                if (not campos[5]) or (campos[5].isdigit() == False): #o campo "numero" precisa ser numerico
                    campos[5] = "0"
                    registro = campos[0]
                    for i in range (1, len(campos)):
                        registro = registro + ";" + campos[i]
                content = content + registro
    
    with open(arquivo_saida, 'w', encoding="utf-8") as out_file:
        out_file.write(content)

def changeEncodingTo(targetEncoding, inputFilePath, outputFilePath):
    encoding = detectEncoding(inputFilePath)
    if(encoding!=targetEncoding):
        with codecs.open(inputFilePath, 'r', encoding) as infile:
            with codecs.open(outputFilePath, 'w', targetEncoding) as outfile:
                for line in infile:
                    outfile.write(line)

def detectEncoding(arquivo):
    with open(arquivo, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

def trimFields(arquivo_entrada, arquivo_saida):
    encoding = detectEncoding(arquivo_entrada)
    #print("encoding: "+encoding)
    content = ""
    with open(arquivo_entrada, 'r', encoding=encoding) as in_file:
        linhas = in_file.readlines()
        content = content + linhas[0]
        for i in range(1, len(linhas)):
            registro = linhas[i]
            campos = registro.split(";")
            content = content + myTrim(campos[0], isKeepSimpleSpace=True)
            for j in range(1, len(campos)):
                content = content + ";" + myTrim(campos[j], isKeepSimpleSpace=True)
    
    with open(arquivo_saida, 'w', encoding="utf-8") as out_file:
        out_file.write(content)

def myTrim(obj, isKeepSimpleSpace=False):
    trimmed = ""
    if isKeepSimpleSpace:
        for o in obj:
            if (o != "   "):
                trimmed = trimmed+o
        return trimmed
    else:
        for o in obj:
            if (o != " ") and (o != "   "):
                trimmed = trimmed+o
        return trimmed

def readHeaders():
    # Nome do arquivo de saída para os cabeçalhos
    completeDatabasePath = '/headers.txt'

    # Lista para armazenar os cabeçalhos
    headers = []

    # Percorre todos os arquivos no diretório
    for filename in os.listdir(rawDataFolder):
        if filename.endswith('.csv'):
            file_path = os.path.join(rawDataFolder, filename)
            with open(file_path, 'r', encoding='latin1') as file:
                # Lê a primeira linha (cabeçalho) e remove a quebra de linha
                header = file.readline().strip()
                headers.append(header)

    # Escreve todos os cabeçalhos no arquivo de saída
    with open(completeDatabasePath, 'w', encoding='latin1') as out_file:
        for header in headers:
            out_file.write(header + '\n')

    print(f'Cabeçalhos salvos em {completeDatabasePath}')

def changeHeaders():
    # Nova coluna a ser adicionada
    new_column = 'LINK_SITE_REDE_SOCIAL'

    # Percorre todos os arquivos no diretório
    for filename in os.listdir(rawDataFolder):
        if filename.endswith('.csv'):
            file_path = os.path.join(rawDataFolder, filename)
            
            # Lê o conteúdo do arquivo
            with open(file_path, 'r', encoding='latin1') as file:
                lines = file.readlines()
            
            # Verifica e modifica o cabeçalho
            header = lines[0].strip()
            if header == "ID_ATRATIVO_TURISTICO;DESCRICAO;CATEGORIA;TIPO_LOGRADOURO;LOGRADOURO;NUMERO;COMPLEMENTO;NOME_BAIRRO_POPULAR;REF_LOCALIZACAO;GEOMETRIA":
                new_header = header.replace('REF_LOCALIZACAO;GEOMETRIA', f'{new_column};REF_LOCALIZACAO;GEOMETRIA')
                lines[0] = new_header + '\n'
                
                # Adiciona uma coluna vazia na posição correta de cada linha de dados
                for i in range(1, len(lines)):
                    columns = lines[i].strip().split(';')
                    columns.insert(-2, '')  # Insere a nova coluna vazia na posição antepenúltima
                    lines[i] = ';'.join(columns) + '\n'
            
            # Escreve o novo conteúdo no arquivo
            with open(file_path, 'w', encoding='latin1') as file:
                file.writelines(lines)

    print('Coluna adicionada e arquivos modificados.')

def removeCSVColumn(arquivo_entrada, arquivo_saida, campo_remover):
  # Lê o arquivo CSV
  encode = detectEncoding(arquivo_entrada)
  df = pd.read_csv(arquivo_entrada, sep=";", encoding=encode)

  # Remove a coluna especificada
  df = df.drop(campo_remover, axis=1)

  # Salva o novo DataFrame em um novo arquivo CSV
  df.to_csv(arquivo_saida, index=False, sep=";", encoding=encode)

def cleanDataBase(output__file):
    removeExcessHeaders(output__file)
    removeLinesNotStartingWithNumber(output__file)
    removeIncompleteLines(output__file, output__file, 11)
    removeIncompatibleLines(output__file, output__file)
    removeAllOccurrences(output__file , "-")
    trimFields(output__file, output__file)

def reduceDataBase(input_file, reducedDataBasePath):
    removeCSVColumn(input_file, reducedDataBasePath, "ID_ATRATIVO_TURISTICO")
    removeCSVColumn(reducedDataBasePath, reducedDataBasePath, "NUMERO")
    removeCSVColumn(reducedDataBasePath, reducedDataBasePath, "LINK_SITE_REDE_SOCIAL")
    removeCSVColumn(reducedDataBasePath, reducedDataBasePath, "GEOMETRIA")

def getAndCleanData():
    readAndUnifyFiles(rawDataFolder, completeDatabasePath)
    cleanDataBase(completeDatabasePath)
    reduceDataBase(completeDatabasePath, reducedDataBasePath)




def cleanText(texto):
    texto = str(texto).lower()  # Converter para minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Remover pontuações
    texto = re.sub(r'\d+', '', texto)  # Remover números
    return texto

""" # Função para tokenizar e remover stopwords
def preProcessText(texto):
    texto = cleanText(texto)
    tokens = word_tokenize(texto)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stop_words] #stop_words é global
    return " ".join(tokens_filtrados) """

def preprocessColumns(dados):
    dados['DESCRICAO_LIMPA'] = dados['DESCRICAO'].apply(preProcessText)
    dados['LOGRADOURO_LIMPO'] = dados['LOGRADOURO'].apply(preProcessText)
    dados['NOME_BAIRRO_POPULAR_LIMPO'] = dados['NOME_BAIRRO_POPULAR'].apply(preProcessText)
    dados['CATEGORIA_LIMPA'] = dados['CATEGORIA'].apply(preProcessText)
    dados['TIPO_LOGRADOURO_LIMPO'] = dados['TIPO_LOGRADOURO'].apply(preProcessText)
    return dados

# Codificar categorias como números (para dados categóricos)
def categorizeColumns(dados):
    label_encoder = LabelEncoder()
    dados['CATEGORIA_CODIFICADA'] = label_encoder.fit_transform(dados['CATEGORIA'])
    dados['TIPO_LOGRADOURO_CODIFICADO'] = label_encoder.fit_transform(dados['TIPO_LOGRADOURO'])
    return dados

def savePreprocessedDatabase(dados, outputFilePath):
    dados_preprocessados = dados[['DESCRICAO_LIMPA', 'LOGRADOURO_LIMPO', 'NOME_BAIRRO_POPULAR_LIMPO', 'CATEGORIA_LIMPA', 'CATEGORIA_CODIFICADA', 'TIPO_LOGRADOURO_LIMPO', 'TIPO_LOGRADOURO_CODIFICADO']]
    dados_preprocessados.to_csv(outputFilePath, index=False)

def preProcessDatabase():
    dados = pd.read_csv(reducedDataBasePath, sep=";") 
    #stop_words = set(stopwords.words('portuguese'))
    dados = preprocessColumns(dados)
    dados = categorizeColumns(dados)
    savePreprocessedDatabase(dados, preprocessedDataPath)




def createWord2Vec(dados):
    # Tokenizar as descrições limpas
    dados['DESCRICAO_TOKENIZADA'] = dados['DESCRICAO_LIMPA'].apply(word_tokenize)
    sentences2 = dados.apply(lambda row: word_tokenize(str(row['LOGRADOURO_LIMPO']) + " " + str(row['NOME_BAIRRO_POPULAR_LIMPO'])), axis=1)
    
    """ for i in sorted(dados['DESCRICAO_TOKENIZADA']):
        print(i) """
    # Criar o modelo Word2Vec
    modelo_descricao_word2vec = Word2Vec(
        sentences=dados['DESCRICAO_TOKENIZADA'],  # Lista de listas de tokens
        vector_size=100,  # Tamanho do vetor de embedding
        window=5,         # Janela de contexto
        min_count=1,      # Frequência mínima para incluir uma palavra
        workers=4,        # Número de threads para treinamento
        sg=0              # Skip-Gram (1) ou CBOW (0)
    )

    modelo_logradouro_word2vec = Word2Vec(
        sentences=sentences2,  # Lista de listas de tokens
        vector_size=100,  # Tamanho do vetor de embedding
        window=5,         # Janela de contexto
        min_count=1,      # Frequência mínima para incluir uma palavra
        workers=4,        # Número de threads para treinamento
        sg=0              # Skip-Gram (1) ou CBOW (0)
    )
    return modelo_descricao_word2vec, modelo_logradouro_word2vec

# Salvar o modelo para uso futuro
def saveModels(modelo_descricao_word2vec, modelo_logradouro_word2vec):
    modelo_descricao_word2vec.save("modelo_descricao_word2vec_turistico.model")
    modelo_logradouro_word2vec.save("modelo_logradouro_word2vec_turistico.model")

# Verificar palavras mais similares a um termo
def testModels(modelo_descricao_word2vec, modelo_logradouro_word2vec):
    print("\n\n(descricao) Palavras similares a 'cultural':", modelo_descricao_word2vec.wv.most_similar('cultural'))
    print("\n")
    print("(logradouro) Palavras similares a 'jardim':", modelo_logradouro_word2vec.wv.most_similar('jardim'))
    print("\n\n")

# Salvar os embeddings como CSV para análises adicionais
def createUnifiedVocab(modelo_descricao_word2vec, modelo_logradouro_word2vec):
    vocabulario_descricao = modelo_descricao_word2vec.wv.index_to_key
    vocabulario_logradouro = modelo_logradouro_word2vec.wv.index_to_key
    vocab_unido = set(vocabulario_descricao).union(set(vocabulario_logradouro))
    return vocab_unido

def createEmbeddings(vocabUnido, modelo_descricao_word2vec, modelo_logradouro_word2vec):
    embeddingsDescricao = []
    embeddingsLogradouro = []
    embedding_combined = []

    for word in vocabUnido:
        # Vetor de descrição, se a palavra estiver no vocabulário do modelo de descrição
        embeddingDescricao = modelo_descricao_word2vec.wv[word] if word in modelo_descricao_word2vec.wv else np.zeros(100)
        embeddingsDescricao.append(embeddingDescricao)

        # Vetor de logradouro, se a palavra estiver no vocabulário do modelo de logradouro
        embeddingLogradouro = modelo_logradouro_word2vec.wv[word] if word in modelo_logradouro_word2vec.wv else np.zeros(100)
        embeddingsLogradouro.append(embeddingLogradouro)

        # Vetor combinado (concatenando os vetores)
        vetor_combined = np.concatenate((embeddingDescricao, embeddingLogradouro))
        embedding_combined.append(vetor_combined)

    embeddings_df = pd.DataFrame({
        "palavra": list(vocabUnido),
        "embedding_descricao": embeddingsDescricao,
        "embedding_logradouro_bairro": embeddingsLogradouro,
        "embedding_combined": embedding_combined
    })
    return embeddings_df

def embeddDataBase():
    dados = pd.read_csv(preprocessedDataPath)
    modelo_descricao, modelo_logradouro = createWord2Vec(dados)
    saveModels(modelo_descricao, modelo_logradouro)
    #testModels(modelo_descricao, modelo_logradouro)
    vocab = createUnifiedVocab(modelo_descricao, modelo_logradouro)
    embedding = createEmbeddings(vocab, modelo_descricao, modelo_logradouro)
    embedding.to_csv(embeddingsPath, index=False)





def loadWord2vecModel(caminho_modelo):
    return gensim.models.Word2Vec.load(caminho_modelo)

# Função para gerar embeddings para um local
def generateEmbeddingPlace(local, modelo_word2vec):
    palavras = local.lower().split()  # Dividir o nome do local em palavras
    embeddings = [modelo_word2vec.wv[palavra] for palavra in palavras if palavra in modelo_word2vec.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)  # Média dos vetores das palavras do local
    else:
        return np.zeros(modelo_word2vec.vector_size)  # Vetor zero se nenhuma palavra estiver no vocabulário

# Função para gerar embeddings a partir das palavras do usuário
def generateEmbeddingUser(preferencias, modelo_word2vec):
    palavras = preferencias.lower().split()
    embeddings_palavras = [modelo_word2vec.wv[palavra] for palavra in palavras if palavra in modelo_word2vec.wv]
    if embeddings_palavras:
        return np.mean(embeddings_palavras, axis=0).reshape(1, -1)
    else:
        #raise ValueError("Nenhuma das palavras do usuário está no vocabulário do modelo.")
        return np.zeros(modelo_word2vec.vector_size).reshape(1, -1)

def recommendPlace(preferencias, locais, modelo_word2vec, top_n=5):
    vetor_usuario = generateEmbeddingUser(preferencias, modelo_word2vec)
    similaridades = {}
    
    for local in locais:
        embedding_local = generateEmbeddingPlace(local, modelo_word2vec)
        similaridade = cosine_similarity(vetor_usuario, embedding_local.reshape(1, -1))[0][0]
        similaridades[local] = similaridade
    
    locais_relevantes = sorted(similaridades.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [local for local, _ in locais_relevantes]

# Função para tokenizar e remover stopwords
def preProcessText(texto):
    texto = cleanText(texto)
    tokens = word_tokenize(texto)
    tokens_filtrados = [toSingular(palavra) for palavra in tokens if palavra not in stop_words] #stop_words é global
    return " ".join(tokens_filtrados)

def toSingular(palavra):
    # Regras comuns de transformação de plural para singular
    regras = [
        (r'ões$', 'ão'),  # balões -> balão
        (r'ães$', 'ão'),  # pães -> pão
        (r'ais$', 'al'),  # animais -> animal
        (r'eis$', 'el'),  # papéis -> papel
        (r'ís$', 'il'),   # fuzis -> fuzil
        (r'ns$', 'm'),    # homens -> homem
        (r'ães$', 'ão'),  # cães -> cão
        (r'ãos$', 'ão'),  # irmãos -> irmão
        (r'^(.*[^s])s$', r'\1'),  # casas -> casa, livros -> livro
    ]

    invariable = [
        'parabens',
        'lapis',
        'virus',
        'atlas',
        'pires',
        'bonus',
        'cais',
        'oculos',
        'onibus',
        'parabéns',
        'lápis',
        'vírus',
        'atlas',
        'pires',
        'bônus',
        'cais',
        'óculos',
        'ônibus'
    ]
    
    palavra = str(palavra).lower()

    if palavra in invariable:
        return palavra
    else:
        for regra, substituicao in regras:
            if re.search(regra, palavra):
                return re.sub(regra, substituicao, palavra)

    return palavra

# Função para recomendar locais com base nas palavras do usuário
def getLocations(preferencias_usuario):
    data = pd.read_csv(reducedDataBasePath, sep=";")
    modelo_word2vec = loadWord2vecModel(inputW2VPath)
    descricoes = sorted(data["DESCRICAO"].dropna().unique()) 
    preferencias = preProcessText(preferencias_usuario)
    try:
        locais_recomendados = recommendPlace(preferencias, descricoes, modelo_word2vec)
    except ValueError as e:
        locais_recomendados = "(Erro ao gerar locais, tente novamente mais tarde)"
        print(e)
    return locais_recomendados




#MAIN

""" print("Getting data and cleaning...")
getAndCleanData()

print("pre-processing for the embedding...")
preProcessDatabase()

print("Starting the word embedding...")
embeddDataBase() """


preferencias_usuario = "Gosto de museus e feiras na Savassi"
print("\n\n\n\n")
print(f"Buscando locais recomendados com base em: '{preferencias_usuario}'")
locais = getLocations(preferencias_usuario)
print("Locais recomendados:")
for local in locais:
    print(local)




print("\n\n\n\n\n\n\n\nThe end!")






































#MAIN




