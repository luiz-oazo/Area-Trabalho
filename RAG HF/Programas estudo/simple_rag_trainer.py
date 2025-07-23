import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from transformers import AutoTokenizer, AutoModel

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

def build_simple_rag_system(urls):
    """Constrói um sistema RAG simplificado usando apenas embeddings e busca."""
    all_documents = []
    print("📚 Coletando documentos...")

    for i, url in enumerate(urls):
        print(f"  Processando {i+1}/{len(urls)}: {url}")
        text = get_article_text(url)
        if text:
            title = extract_title_from_url(url)
            all_documents.append({"title": title, "text": text, "url": url})

    if not all_documents:
        print("❌ Nenhum documento coletado.")
        return None, None, None

    print(f"✅ Coletados {len(all_documents)} documentos")

    # Criar embeddings
    print("🔍 Criando embeddings...")
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    # Dividir textos em chunks menores para melhor performance
    chunks = []
    chunk_metadata = []

    for doc in all_documents:
        text = doc['text']
        # Dividir em chunks de ~500 caracteres
        chunk_size = 500
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip()) > 50:  # Ignorar chunks muito pequenos
                chunks.append(chunk)
                chunk_metadata.append({
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_id': len(chunks) - 1,
                    'start_pos': i
                })

    print(f"📊 Criados {len(chunks)} chunks de texto")

    # Criar embeddings para os chunks
    embeddings = embedding_model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Criar índice Faiss
    print("🔗 Construindo índice Faiss...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Criar dataset
    dataset_dict = {
        'chunks': chunks,
        'metadata': chunk_metadata,
        'embeddings': embeddings.tolist()
    }

    return embedding_model, index, dataset_dict

def save_simple_rag_system(embedding_model, index, dataset_dict, save_dir="/content/simple_rag_system"):
    """Salva o sistema RAG simplificado."""
    print(f"💾 Salvando sistema em: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Salvar modelo de embedding
    print("  Salvando modelo de embedding...")
    embedding_model.save(f"{save_dir}/embedding_model")

    # Salvar índice Faiss
    print("  Salvando índice Faiss...")
    faiss.write_index(index, f"{save_dir}/faiss_index.bin")

    # Salvar dataset
    print("  Salvando dataset...")
    with open(f"{save_dir}/dataset.pkl", "wb") as f:
        pickle.dump(dataset_dict, f)

    # Salvar metadados
    metadata = {
        "num_chunks": len(dataset_dict['chunks']),
        "embedding_dim": index.d,
        "model_name": "sentence-transformers/all-mpnet-base-v2"
    }

    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Sistema salvo com sucesso!")
    print(f"📁 Arquivos salvos:")
    for file in os.listdir(save_dir):
        print(f"  - {file}")

# --- Execução Principal ---
if __name__ == "__main__":
    print("🚀 CRIANDO SISTEMA RAG SIMPLIFICADO")
    print("="*50)

    # URLs de exemplo (você pode modificar)
    starting_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    print(f"🔍 Buscando links de: {starting_url}")
    wikipedia_urls = get_wikipedia_links(starting_url)
    print(f"📋 Encontrados {len(wikipedia_urls)} links")

    # Usar apenas 8 URLs para ser mais rápido
    num_urls = min(8, len(wikipedia_urls))
    wikipedia_urls = wikipedia_urls[:num_urls]
    print(f"📚 Usando {len(wikipedia_urls)} links")

    # Construir sistema
    print("\n🔧 CONSTRUINDO SISTEMA...")
    print("-"*30)
    embedding_model, index, dataset_dict = build_simple_rag_system(wikipedia_urls)

    if embedding_model and index and dataset_dict:
        print("\n✅ Sistema construído com sucesso!")

        # Salvar sistema
        print("\n💾 SALVANDO...")
        print("-"*30)
        save_simple_rag_system(embedding_model, index, dataset_dict)

        print("\n🎉 CONCLUÍDO!")
        print("="*50)
        print("Use o arquivo de consulta para fazer perguntas!")

    else:
        print("❌ Falha ao construir sistema.")
