import requests
from bs4 import BeautifulSoup
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from transformers import BartForConditionalGeneration, DPRQuestionEncoder
import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import pickle
import os
import json

def extract_title_from_url(url):
    """Extrai o título da URL do Wikipedia."""
    return url.split("/wiki/")[-1].replace("_", " ")

def get_wikipedia_links(url):
    """Busca todos os links internos do Wikipedia de uma página."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/wiki/') and ':' not in href:
                full_url = f"https://en.wikipedia.org{href}"
                links.add(full_url)
        return list(links)
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar URL {url}: {e}")
        return []

def get_article_text(url):
    """Busca e limpa o conteúdo de texto de um artigo do Wikipedia."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove elementos indesejados
        for tag in soup.find_all(['table', 'sup', 'span'], class_=['infobox', 'reference']):
             tag.decompose()
        text = ""
        for paragraph in soup.find_all('p'):
            text += paragraph.get_text() + "\n"
        return text.strip()
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar texto do artigo {url}: {e}")
        return ""

def build_rag_system(urls):
    """Constrói um sistema RAG usando Faiss para indexação."""
    all_documents = []
    print("📚 Coletando documentos...")

    for i, url in enumerate(urls):
        print(f"  Processando {i+1}/{len(urls)}: {url}")
        text = get_article_text(url)
        if text:
            title = extract_title_from_url(url)
            all_documents.append({"title": title, "text": text, "url": url})

    if not all_documents:
        print("❌ Nenhum documento coletado. Não é possível construir o sistema RAG.")
        return None, None, None, None, None

    print(f"✅ Coletados {len(all_documents)} documentos")

    # Criar Dataset do Hugging Face
    print("📊 Criando dataset...")
    dataset = Dataset.from_list(all_documents)

    # Inicializar componentes RAG
    print("🤖 Inicializando tokenizer RAG...")
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

    # Criar índice Faiss
    print("🔍 Construindo índice Faiss...")
    # Obter embeddings para os documentos
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = embedding_model.encode([doc['text'] for doc in all_documents])
    embeddings = np.array(embeddings).astype('float32')

    # Converter embeddings para lista de listas
    embeddings_list = [e.tolist() for e in embeddings]

    # Adicionar embeddings como nova coluna ao dataset
    dataset = dataset.add_column("embeddings", embeddings_list)

    # Construir índice Faiss
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Adicionar índice Faiss ao dataset
    dataset.add_faiss_index(column='embeddings', custom_index=index)

    # Inicializar o retriever com o dataset
    print("🔗 Inicializando retriever...")
    retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", indexed_dataset=dataset)

    # Inicializar modelo RAG
    print("🧠 Inicializando modelo RAG...")
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever)

    return tokenizer, rag_model, dataset, index, embedding_model

def save_rag_system(tokenizer, rag_model, dataset, index, embedding_model, save_dir="rag_system"):
    """Salva todos os componentes do sistema RAG de forma correta."""
    print(f"💾 Salvando sistema RAG em: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Salvar tokenizer
    print("  Salvando tokenizer...")
    tokenizer.save_pretrained(f"{save_dir}/tokenizer")

    # Salvar modelo RAG completo (IMPORTANTE: salvar como um todo)
    print("  Salvando modelo RAG completo...")
    rag_model.save_pretrained(f"{save_dir}/model")

    # Salvar dataset limpo (sem índices Faiss)
    print("  Salvando dataset...")
    try:
        # Criar dataset limpo sem índices
        clean_dataset = Dataset.from_dict({
            'title': dataset['title'],
            'text': dataset['text'],
            'url': dataset['url'],
            'embeddings': dataset['embeddings']
        })
        clean_dataset.save_to_disk(f"{save_dir}/dataset")
    except Exception as e:
        print(f"    Erro ao salvar dataset: {e}")
        print("    Salvando como pickle...")
        with open(f"{save_dir}/dataset.pkl", "wb") as f:
            pickle.dump(dataset.to_dict(), f)

    # Salvar índice Faiss
    print("  Salvando índice Faiss...")
    faiss.write_index(index, f"{save_dir}/faiss_index.bin")

    # Salvar modelo de embedding
    print("  Salvando modelo de embedding...")
    embedding_model.save(f"{save_dir}/embedding_model")

    # Salvar configuração do retriever
    print("  Salvando configuração do retriever...")
    retriever_config = {
        "model_name": "facebook/rag-sequence-nq",
        "dataset_info": {
            "num_documents": len(dataset),
            "embedding_dim": index.d,
            "columns": dataset.column_names
        }
    }

    with open(f"{save_dir}/retriever_config.json", "w") as f:
        json.dump(retriever_config, f, indent=2)

    # Salvar metadados
    print("  Salvando metadados...")
    metadata = {
        "feedback_scores": {},
        "document_scores": {},
        "num_documents": len(dataset),
        "embedding_dim": index.d,
        "model_type": "rag-sequence-nq",
        "embedding_model": "sentence-transformers/all-mpnet-base-v2"
    }

    with open(f"{save_dir}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Sistema RAG salvo com sucesso em: {save_dir}")
    print(f"📁 Arquivos salvos:")
    for root, dirs, files in os.walk(save_dir):
        level = root.replace(save_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")

# --- Execução Principal ---
if __name__ == "__main__":
    print("🚀 INICIANDO TREINAMENTO DO SISTEMA RAG")
    print("="*50)

    # Definir URL inicial do Wikipedia
    starting_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    print(f"🔍 Buscando links de: {starting_url}")
    wikipedia_urls = get_wikipedia_links(starting_url)
    print(f"📋 Encontrados {len(wikipedia_urls)} links")

    # Limitar número de URLs para demonstração
    num_urls = min(10, len(wikipedia_urls))  # Reduzido para 10 para ser mais rápido
    wikipedia_urls = wikipedia_urls[:num_urls]
    print(f"📚 Usando {len(wikipedia_urls)} links para construir o sistema RAG")

    # Construir sistema RAG
    print("\n🔧 CONSTRUINDO SISTEMA RAG...")
    print("-"*30)
    tokenizer, rag_model, dataset, faiss_index, embedding_model = build_rag_system(wikipedia_urls)

    if tokenizer and rag_model and dataset and faiss_index and embedding_model:
        print("\n✅ Sistema RAG construído com sucesso!")

        # Salvar sistema RAG
        print("\n💾 SALVANDO SISTEMA...")
        print("-"*30)
        save_rag_system(tokenizer, rag_model, dataset, faiss_index, embedding_model)

        print("\n🎉 TREINAMENTO CONCLUÍDO!")
        print("="*50)
        print("Agora você pode usar o sistema de consulta!")

    else:
        print("❌ Falha ao construir sistema RAG. Verifique os logs acima.")
