import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
import torch
from tqdm import tqdm

def extract_title_from_url(url):
    """Extrai o título da URL do Wikipedia."""
    return url.split("/wiki/")[-1].replace("_", " ")

def get_article_text(url):
    """Busca e limpa o conteúdo de texto de um artigo do Wikipedia."""
    try:
        response = requests.get(url, timeout=15)
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
        print(f"    ⚠️ Erro ao buscar {url}: {e}")
        return ""

def build_final_rag_system(urls):
    """Constrói sistema RAG final sem problemas de segfault."""
    all_documents = []
    print("📚 Coletando documentos...")

    for i, url in enumerate(urls):
        print(f"  Processando {i+1}/{len(urls)}: {extract_title_from_url(url)}")
        text = get_article_text(url)
        if text and len(text) > 200:  # Garantir conteúdo suficiente
            title = extract_title_from_url(url)
            all_documents.append({"title": title, "text": text, "url": url})
        else:
            print(f"    ⚠️ Documento muito pequeno, pulando...")

    if not all_documents:
        print("❌ Nenhum documento coletado.")
        return None, None, None

    print(f"✅ Coletados {len(all_documents)} documentos válidos")

    # Configurar ambiente para evitar problemas
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Inicializar modelo de embedding de forma mais segura
    print("🔍 Inicializando modelo de embedding...")
    try:
        # Usar modelo menor e mais estável
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✅ Modelo carregado com sucesso")
    except Exception as e:
        print(f"  ❌ Erro ao carregar modelo: {e}")
        return None, None, None

    # Criar chunks menores e mais gerenciáveis
    print("📊 Criando chunks de texto...")
    chunks = []
    chunk_metadata = []

    for doc_idx, doc in enumerate(all_documents):
        text = doc['text']
        # Chunks menores para evitar problemas
        chunk_size = 300
        overlap = 50

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size].strip()
            if len(chunk) > 80:  # Chunks mínimos
                chunks.append(chunk)
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

    # Criar embeddings de forma mais segura
    print("🔗 Gerando embeddings...")
    try:
        # Processar em lotes muito pequenos
        batch_size = 16
        all_embeddings = []

        # Usar tqdm para mostrar progresso
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processando embeddings"):
            batch = chunks[i:i+batch_size]

            # Método correto para gerar embeddings
            batch_embeddings = embedding_model.encode(
                batch, 
                batch_size=8,  # Lote ainda menor
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
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
        # Usar índice mais simples
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        print(f"✅ Índice Faiss criado com {index.ntotal} vetores de dimensão {dimension}")
    except Exception as e:
        print(f"❌ Erro ao criar índice Faiss: {e}")
        return None, None, None

    # Criar dataset
    dataset_dict = {
        'chunks': chunks,
        'metadata': chunk_metadata,
        'documents': all_documents,
        'num_chunks': len(chunks),
        'num_docs': len(all_documents)
    }

    return embedding_model, index, dataset_dict

def save_final_rag_system(embedding_model, index, dataset_dict, save_dir="/content/simple_rag_system"):
    """Salva o sistema RAG final."""
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

        # Salvar informações do sistema
        info = {
            "num_chunks": dataset_dict['num_chunks'],
            "num_documents": dataset_dict['num_docs'],
            "embedding_dim": index.d,
            "model_name": "all-MiniLM-L6-v2",
            "index_type": "IndexFlatL2",
            "created_successfully": True
        }

        with open(f"{save_dir}/system_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"✅ Sistema salvo com sucesso!")
        print(f"📁 Arquivos criados:")
        for file in os.listdir(save_dir):
            filepath = os.path.join(save_dir, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  - {file} ({size/1024:.1f} KB)")

        return True

    except Exception as e:
        print(f"❌ Erro ao salvar sistema: {e}")
        return False

# --- Execução Principal ---
if __name__ == "__main__":
    print("🚀 SISTEMA RAG FINAL - SEM SEGFAULTS")
    print("="*50)

    # URLs específicas e confiáveis sobre IA
    ai_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning", 
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Neural_network"
    ]

    print("📚 Processando artigos sobre IA:")
    for i, url in enumerate(ai_urls, 1):
        title = extract_title_from_url(url)
        print(f"  {i}. {title}")

    # Construir sistema
    print("\n🔧 CONSTRUINDO SISTEMA...")
    print("-"*30)

    embedding_model, index, dataset_dict = build_final_rag_system(ai_urls)

    if embedding_model and index and dataset_dict:
        print("\n✅ Sistema construído com sucesso!")
        print(f"📊 Estatísticas:")
        print(f"  - Documentos: {dataset_dict['num_docs']}")
        print(f"  - Chunks: {dataset_dict['num_chunks']}")
        print(f"  - Dimensão embeddings: {index.d}")

        # Salvar sistema
        print("\n💾 SALVANDO...")
        print("-"*30)
        if save_final_rag_system(embedding_model, index, dataset_dict):
            print("\n🎉 SISTEMA PRONTO!")
            print("="*50)
            print("✅ Treinamento concluído com sucesso!")
            print("🚀 Execute agora: python smart_rag_with_feedback_fixed.py")
            print("\n💡 O sistema está pronto para:")
            print("  - Responder perguntas sobre IA")
            print("  - Aprender com seu feedback")
            print("  - Melhorar continuamente")
        else:
            print("❌ Erro ao salvar sistema.")

    else:
        print("❌ Falha ao construir sistema.")
        print("💡 Tente verificar sua conexão com a internet.")
