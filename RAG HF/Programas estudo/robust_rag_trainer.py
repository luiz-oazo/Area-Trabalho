import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
import torch

def extract_title_from_url(url):
    """Extrai o título da URL do Wikipedia."""
    return url.split("/wiki/")[-1].replace("_", " ")

def get_wikipedia_links(url):
    """Busca todos os links internos do Wikipedia de uma página."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        links = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('/wiki/') and ':' not in href:
                full_url = f"https://en.wikipedia.org{href}"
                links.add(full_url)
        return list(links)
    except Exception as e:
        print(f"Erro ao buscar URL {url}: {e}")
        return []

def get_article_text(url):
    """Busca e limpa o conteúdo de texto de um artigo do Wikipedia."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove elementos indesejados
        for tag in soup.find_all(['table', 'sup', 'span', 'div'], 
                                class_=['infobox', 'reference', 'navbox', 'sidebar']):
            tag.decompose()

        # Remove scripts e styles
        for script in soup(["script", "style"]):
            script.decompose()

        text = ""
        for paragraph in soup.find_all('p'):
            para_text = paragraph.get_text()
            if len(para_text.strip()) > 20:  # Ignorar parágrafos muito pequenos
                text += para_text + "\n"

        return text.strip()
    except Exception as e:
        print(f"Erro ao buscar texto do artigo {url}: {e}")
        return ""

def build_robust_rag_system(urls):
    """Constrói um sistema RAG mais robusto."""
    all_documents = []
    print("📚 Coletando documentos...")

    for i, url in enumerate(urls):
        print(f"  Processando {i+1}/{len(urls)}: {url}")
        text = get_article_text(url)
        if text and len(text) > 100:  # Garantir que há conteúdo suficiente
            title = extract_title_from_url(url)
            all_documents.append({"title": title, "text": text, "url": url})
        else:
            print(f"    ⚠️ Documento muito pequeno ou vazio, pulando...")

    if not all_documents:
        print("❌ Nenhum documento coletado.")
        return None, None, None

    print(f"✅ Coletados {len(all_documents)} documentos válidos")

    # Criar embeddings com configurações mais seguras
    print("🔍 Inicializando modelo de embedding...")

    # Configurar para evitar problemas de multiprocessing
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Tentar usar GPU se disponível, senão CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Usando dispositivo: {device}")

        embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)

        # Desabilitar multiprocessing para evitar segfaults
        embedding_model.encode = lambda texts, **kwargs: embedding_model._encode(texts, show_progress_bar=True, batch_size=16, **kwargs)

    except Exception as e:
        print(f"  Erro ao carregar modelo: {e}")
        print("  Tentando modelo alternativo...")
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Dividir textos em chunks menores
    print("📊 Criando chunks de texto...")
    chunks = []
    chunk_metadata = []

    for doc_idx, doc in enumerate(all_documents):
        text = doc['text']
        # Dividir em chunks de ~400 caracteres com overlap
        chunk_size = 400
        overlap = 50

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            if len(chunk.strip()) > 100:  # Ignorar chunks muito pequenos
                chunks.append(chunk.strip())
                chunk_metadata.append({
                    'title': doc['title'],
                    'url': doc['url'],
                    'chunk_id': len(chunks) - 1,
                    'doc_id': doc_idx,
                    'start_pos': i
                })

    print(f"📊 Criados {len(chunks)} chunks de texto")

    if not chunks:
        print("❌ Nenhum chunk criado.")
        return None, None, None

    # Criar embeddings em lotes menores para evitar problemas de memória
    print("🔗 Gerando embeddings...")
    try:
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"  Processando lote {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

            # Gerar embeddings para este lote
            batch_embeddings = embedding_model.encode(
                batch, 
                show_progress_bar=False,
                batch_size=16,
                convert_to_numpy=True
            )
            all_embeddings.append(batch_embeddings)

        # Concatenar todos os embeddings
        embeddings = np.vstack(all_embeddings).astype('float32')
        print(f"✅ Embeddings criados: {embeddings.shape}")

    except Exception as e:
        print(f"❌ Erro ao criar embeddings: {e}")
        return None, None, None

    # Criar índice Faiss
    print("🔗 Construindo índice Faiss...")
    try:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print(f"✅ Índice Faiss criado com {index.ntotal} vetores")
    except Exception as e:
        print(f"❌ Erro ao criar índice Faiss: {e}")
        return None, None, None

    # Criar dataset
    dataset_dict = {
        'chunks': chunks,
        'metadata': chunk_metadata,
        'embeddings': embeddings.tolist(),
        'documents': all_documents
    }

    return embedding_model, index, dataset_dict

def save_robust_rag_system(embedding_model, index, dataset_dict, save_dir="/content/simple_rag_system"):
    """Salva o sistema RAG de forma robusta."""
    print(f"💾 Salvando sistema em: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    try:
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
            "num_documents": len(dataset_dict['documents']),
            "embedding_dim": index.d,
            "model_name": "sentence-transformers/all-mpnet-base-v2",
            "created_at": "2024-01-01"
        }

        with open(f"{save_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Sistema salvo com sucesso!")
        print(f"📁 Arquivos salvos:")
        for file in os.listdir(save_dir):
            size = os.path.getsize(os.path.join(save_dir, file))
            print(f"  - {file} ({size/1024/1024:.1f} MB)")

        return True

    except Exception as e:
        print(f"❌ Erro ao salvar sistema: {e}")
        return False

# --- Execução Principal ---
if __name__ == "__main__":
    print("🚀 CRIANDO SISTEMA RAG ROBUSTO")
    print("="*50)

    # URLs de exemplo mais focadas em IA
    ai_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning", 
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network",
        "https://en.wikipedia.org/wiki/Natural_language_processing"
    ]

    print("📚 Usando URLs focadas em IA:")
    for i, url in enumerate(ai_urls, 1):
        print(f"  {i}. {url.split('/')[-1].replace('_', ' ')}")

    # Construir sistema
    print("\n🔧 CONSTRUINDO SISTEMA...")
    print("-"*30)
    embedding_model, index, dataset_dict = build_robust_rag_system(ai_urls)

    if embedding_model and index and dataset_dict:
        print("\n✅ Sistema construído com sucesso!")

        # Salvar sistema
        print("\n💾 SALVANDO...")
        print("-"*30)
        if save_robust_rag_system(embedding_model, index, dataset_dict):
            print("\n🎉 CONCLUÍDO!")
            print("="*50)
            print("✅ Sistema RAG pronto para uso!")
            print("Execute agora: python smart_rag_with_feedback_fixed.py")
        else:
            print("❌ Erro ao salvar sistema.")

    else:
        print("❌ Falha ao construir sistema.")
