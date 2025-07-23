#!/usr/bin/env python3
"""
PROGRAMA 1: TREINAMENTO RAG
Cria e treina o sistema RAG com artigos sobre IA
"""

import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
import json
from tqdm import tqdm
import sys
import warnings
import torch

# Suprimir warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

class RAGTrainer:
    def __init__(self, save_dir="./rag_system"):  # Corrigido: mudança de diretório
        self.save_dir = save_dir
        self.articles = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning", 
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Neural_network",
            "https://en.wikipedia.org/wiki/Natural_language_processing"
        ]

    def get_article_content(self, url):
        """Extrai conteúdo limpo de um artigo do Wikipedia."""
        try:
            print(f"  Baixando: {url.split('/')[-1].replace('_', ' ')}")  # Corrigido: removido emoji
            response = requests.get(url, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove elementos desnecessários
            for element in soup.find_all(['table', 'sup', 'span', 'div'], 
                                       class_=['infobox', 'reference', 'navbox']):
                element.decompose()

            # Extrai texto dos parágrafos
            text = ""
            for p in soup.find_all('p'):
                paragraph = p.get_text().strip()
                if len(paragraph) > 30:  # Ignora parágrafos muito pequenos
                    text += paragraph + "\n\n"

            return {
                'title': url.split('/')[-1].replace('_', ' '),
                'url': url,
                'content': text.strip()
            }

        except Exception as e:
            print(f"    Erro ao baixar {url}: {e}")  # Corrigido: removido emoji
            return None

    def create_chunks(self, documents):
        """Divide documentos em chunks menores."""
        print("\nCriando chunks de texto...")  # Corrigido: removido emoji
        chunks = []
        metadata = []

        for doc_id, doc in enumerate(documents):
            content = doc['content']
            chunk_size = 400
            overlap = 80

            # Divide em chunks com overlap
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size].strip()
                if len(chunk) > 100:  # Chunks mínimos
                    chunks.append(chunk)
                    metadata.append({
                        'doc_id': doc_id,
                        'title': doc['title'],
                        'url': doc['url'],
                        'chunk_id': len(chunks) - 1,
                        'start_pos': i
                    })

        print(f"  Criados {len(chunks)} chunks de {len(documents)} documentos")  # Corrigido: removido emoji
        return chunks, metadata

    def create_embeddings(self, chunks):
        """Cria embeddings dos chunks usando SentenceTransformer."""
        print("\nGerando embeddings...")  # Corrigido: removido emoji

        # Configurações para suprimir warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            # Modelo otimizado para CPU
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"  Modelo carregado: all-MiniLM-L6-v2")  # Corrigido: removido emoji

            # Gera embeddings em lotes pequenos
            batch_size = 32
            embeddings = []

            for i in tqdm(range(0, len(chunks), batch_size), 
                         desc="  Processando", unit="lote"):
                batch = chunks[i:i + batch_size]
                batch_emb = model.encode(
                    batch,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.append(batch_emb)

            # Concatena todos os embeddings
            all_embeddings = np.vstack(embeddings).astype('float32')
            print(f"  Embeddings criados: {all_embeddings.shape}")  # Corrigido: removido emoji

            return model, all_embeddings

        except Exception as e:
            print(f"  Erro ao criar embeddings: {e}")  # Corrigido: removido emoji
            return None, None

    def create_faiss_index(self, embeddings):
        """Cria índice Faiss para busca rápida."""
        print("\nConstruindo índice Faiss...")  # Corrigido: removido emoji
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            print(f"  Índice criado: {index.ntotal} vetores, dimensão {dimension}")  # Corrigido: removido emoji
            return index
        except Exception as e:
            print(f"  Erro ao criar índice: {e}")  # Corrigido: removido emoji
            return None

    def save_system(self, model, index, chunks, metadata, documents):
        """Salva todo o sistema RAG."""
        print(f"\nSalvando sistema em: {os.path.abspath(self.save_dir)}")  # Corrigido: removido emoji

        try:
            os.makedirs(self.save_dir, exist_ok=True)

            # Salva modelo de embedding
            print("  Salvando modelo de embedding...")  # Corrigido: removido emoji
            model.save(f"{self.save_dir}/embedding_model")

            # Salva índice Faiss
            print("  Salvando índice Faiss...")  # Corrigido: removido emoji
            faiss.write_index(index, f"{self.save_dir}/faiss_index.bin")

            # Salva dados - Corrigido: conversão de tipos numpy
            print("  Salvando dados...")  # Corrigido: removido emoji

            # Converter metadados para tipos nativos do Python
            metadata_clean = []
            for meta in metadata:
                meta_clean = {}
                for key, value in meta.items():
                    if isinstance(value, (np.int64, np.int32)):
                        meta_clean[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        meta_clean[key] = float(value)
                    else:
                        meta_clean[key] = value
                metadata_clean.append(meta_clean)

            data = {
                'chunks': chunks,
                'metadata': metadata_clean,  # Usar metadados limpos
                'documents': documents,
                'stats': {
                    'num_chunks': len(chunks),
                    'num_documents': len(documents),
                    'embedding_dim': int(index.d)  # Converter para int nativo
                }
            }

            with open(f"{self.save_dir}/rag_data.pkl", "wb") as f:
                pickle.dump(data, f)

            # Salva informações do sistema - Corrigido: conversão de tipos
            info = {
                'created_at': '2024-01-01',
                'model_name': 'all-MiniLM-L6-v2',
                'num_chunks': len(chunks),
                'num_documents': len(documents),
                'embedding_dimension': int(index.d),  # Converter para int nativo
                'system_ready': True
            }

            with open(f"{self.save_dir}/system_info.json", "w") as f:
                json.dump(info, f, indent=2)

            print("  Sistema salvo com sucesso!")  # Corrigido: removido emoji

            # Mostra arquivos criados
            print("\nArquivos criados:")  # Corrigido: removido emoji
            for file in os.listdir(self.save_dir):
                path = os.path.join(self.save_dir, file)
                if os.path.isfile(path):
                    size = os.path.getsize(path) / 1024
                    print(f"  - {file} ({size:.1f} KB)")

            return True

        except Exception as e:
            print(f"  Erro ao salvar: {e}")  # Corrigido: removido emoji
            return False

    def train(self):
        """Executa o treinamento completo."""
        print("INICIANDO TREINAMENTO RAG")  # Corrigido: removido emoji
        print("=" * 50)
        print("Otimizado para sistema")  # Corrigido: removido emoji e texto
        print("Foco em Inteligência Artificial")  # Corrigido: removido emoji
        print("=" * 50)

        # 1. Coleta documentos
        print("\nCOLETANDO DOCUMENTOS...")  # Corrigido: removido emoji
        documents = []
        for url in self.articles:
            doc = self.get_article_content(url)
            if doc:
                documents.append(doc)

        if not documents:
            print("Nenhum documento coletado!")  # Corrigido: removido emoji
            return False

        print(f"{len(documents)} documentos coletados")  # Corrigido: removido emoji

        # 2. Cria chunks
        chunks, metadata = self.create_chunks(documents)
        if not chunks:
            print("Nenhum chunk criado!")  # Corrigido: removido emoji
            return False

        # 3. Cria embeddings
        model, embeddings = self.create_embeddings(chunks)
        if model is None or embeddings is None:
            print("Falha ao criar embeddings!")  # Corrigido: removido emoji
            return False

        # 4. Cria índice
        index = self.create_faiss_index(embeddings)
        if index is None:
            print("Falha ao criar índice!")  # Corrigido: removido emoji
            return False

        # 5. Salva sistema
        if self.save_system(model, index, chunks, metadata, documents):
            print("\nTREINAMENTO CONCLUÍDO!")  # Corrigido: removido emoji
            print("=" * 50)
            print("Sistema RAG pronto para uso")  # Corrigido: removido emoji
            print("Execute agora: python 2_rag_query_smart.py")  # Corrigido: removido emoji
            print("Ou execute: python 3_rag_feedback_memory.py")  # Corrigido: removido emoji
            return True
        else:
            print("Falha ao salvar sistema!")  # Corrigido: removido emoji
            return False

def main():
    """Função principal."""
    print("RAG TRAINER")  # Corrigido: removido emoji e texto específico
    print("Pressione Ctrl+C para cancelar a qualquer momento\n")

    try:
        trainer = RAGTrainer()
        success = trainer.train()

        if success:
            print("\nPronto! Agora você pode usar os outros programas.")  # Corrigido: removido emoji
        else:
            print("\nTreinamento falhou. Verifique sua conexão com a internet.")  # Corrigido: removido emoji

    except KeyboardInterrupt:
        print("\n\nTreinamento cancelado pelo usuário.")  # Corrigido: removido emoji
        sys.exit(0)
    except Exception as e:
        print(f"\nErro inesperado: {e}")  # Corrigido: removido emoji
        sys.exit(1)

if __name__ == "__main__":
    main()
