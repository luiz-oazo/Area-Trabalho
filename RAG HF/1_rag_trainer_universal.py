#!/usr/bin/env python3
"""
PROGRAMA 1: TREINAMENTO RAG UNIVERSAL
Funciona com Wikipedia, blogs, sites de notícias, documentação, etc.
Versão corrigida e otimizada
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
import time
from urllib.parse import urlparse
import warnings
import torch

# Suprimir warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

class UniversalRAGTrainer:
    def __init__(self, save_dir="./rag_system"):  # Corrigido: diretório alterado
        self.save_dir = save_dir

        # CONFIGURE SUAS URLs AQUI - QUALQUER SITE!
        self.articles = [
            # Exemplos de diferentes tipos de sites:

            # Wikipedia (sempre funciona bem)
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",

            # Sites de notícias (exemplo)
            # "https://www.bbc.com/news/technology-51064369",

            # Blogs técnicos (exemplo)
            # "https://openai.com/blog/chatgpt",

            # Documentação (exemplo)
            # "https://docs.python.org/3/tutorial/introduction.html",

            # Sites educacionais (exemplo)
            # "https://www.khanacademy.org/computing/intro-to-programming",

            # Adicione suas URLs aqui:
            "https://en.wikipedia.org/wiki/Deep_learning",
            "https://en.wikipedia.org/wiki/Neural_network",
            "https://en.wikipedia.org/wiki/Natural_language_processing"
        ]

        # Headers para parecer um navegador real
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def extract_text_universal(self, soup, url):
        """Extrai texto de qualquer tipo de site."""
        try:
            # Remove elementos indesejados comuns
            unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 
                           'aside', 'advertisement', 'ads', 'sidebar']

            for tag in unwanted_tags:
                for element in soup.find_all(tag):
                    element.decompose()

            # Remove classes/IDs comuns de elementos indesejados
            unwanted_classes = ['advertisement', 'ads', 'sidebar', 'nav', 
                              'header', 'footer', 'menu', 'social', 'share',
                              'comment', 'related', 'popup', 'modal']

            for class_name in unwanted_classes:
                for element in soup.find_all(class_=class_name):
                    element.decompose()
                for element in soup.find_all(id=class_name):
                    element.decompose()

            # Estratégias de extração por prioridade
            text_content = ""

            # 1. Tenta encontrar conteúdo principal
            main_selectors = [
                'main', 'article', '.content', '.post', '.entry',
                '.article-body', '.post-content', '.entry-content',
                '#content', '#main', '#article'
            ]

            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    text_content = main_content.get_text(separator='\n', strip=True)
                    if len(text_content) > 500:  # Conteúdo substancial
                        break

            # 2. Se não encontrou conteúdo principal, tenta parágrafos
            if len(text_content) < 500:
                paragraphs = soup.find_all('p')
                text_parts = []
                for p in paragraphs:
                    p_text = p.get_text().strip()
                    if len(p_text) > 30:  # Ignora parágrafos muito pequenos
                        text_parts.append(p_text)
                text_content = '\n\n'.join(text_parts)

            # 3. Se ainda não tem conteúdo, tenta divs com texto
            if len(text_content) < 500:
                divs = soup.find_all('div')
                text_parts = []
                for div in divs:
                    # Ignora divs com muitos elementos filhos (provavelmente layout)
                    if len(div.find_all()) < 5:
                        div_text = div.get_text().strip()
                        if len(div_text) > 50:
                            text_parts.append(div_text)
                text_content = '\n\n'.join(text_parts[:20])  # Limita a 20 divs

            # 4. Último recurso: todo o texto da página
            if len(text_content) < 200:
                text_content = soup.get_text(separator='\n', strip=True)

            # Limpa o texto
            lines = text_content.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith(('©', 'Cookie', 'Privacy')):
                    clean_lines.append(line)

            final_text = '\n\n'.join(clean_lines)

            return final_text[:50000]  # Limita a 50k caracteres

        except Exception as e:
            print(f"    Erro na extração de texto: {e}")
            return ""

    def get_article_content_universal(self, url):
        """Extrai conteúdo de qualquer site."""
        try:
            domain = urlparse(url).netloc
            print(f"  Baixando: {domain}")

            # Faz requisição com headers de navegador
            response = requests.get(url, headers=self.headers, timeout=20)
            response.raise_for_status()

            # Detecta encoding
            response.encoding = response.apparent_encoding

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extrai texto usando método universal
            text = self.extract_text_universal(soup, url)

            if len(text) < 100:
                print(f"    Pouco conteúdo extraído de {domain}")
                return None

            # Tenta extrair título
            title = ""
            title_selectors = ['h1', 'title', '.title', '.headline', '.post-title']
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()[:100]
                    break

            if not title:
                title = domain

            print(f"    Extraído: {len(text)} caracteres")

            return {
                'title': title,
                'url': url,
                'domain': domain,
                'content': text.strip()
            }

        except requests.exceptions.RequestException as e:
            print(f"    Erro de conexão com {url}: {e}")
            return None
        except Exception as e:
            print(f"    Erro ao processar {url}: {e}")
            return None

    def create_chunks(self, documents):
        """Divide documentos em chunks menores."""
        print("\nCriando chunks de texto...")
        chunks = []
        metadata = []

        for doc_id, doc in enumerate(documents):
            content = doc['content']
            chunk_size = 500  # Chunks maiores para sites diversos
            overlap = 100

            # Divide em chunks com overlap
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i:i + chunk_size].strip()
                if len(chunk) > 150:  # Chunks mínimos maiores
                    chunks.append(chunk)
                    metadata.append({
                        'doc_id': doc_id,
                        'title': doc['title'],
                        'url': doc['url'],
                        'domain': doc['domain'],
                        'chunk_id': len(chunks) - 1,
                        'start_pos': i
                    })

        print(f"  Criados {len(chunks)} chunks de {len(documents)} documentos")
        return chunks, metadata

    def create_embeddings(self, chunks):
        """Cria embeddings dos chunks usando SentenceTransformer."""
        print("\nGerando embeddings...")

        # Configurações para suprimir warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        try:
            # Modelo otimizado para CPU
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print(f"  Modelo carregado: all-MiniLM-L6-v2")

            # Gera embeddings em lotes pequenos
            batch_size = 16  # Menor para sites diversos
            embeddings = []

            for i in tqdm(range(0, len(chunks), batch_size), 
                         desc="  Processando", unit="lote"):
                batch = chunks[i:i + batch_size]
                batch_emb = model.encode(
                    batch,
                    batch_size=8,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.append(batch_emb)

            # Concatena todos os embeddings
            all_embeddings = np.vstack(embeddings).astype('float32')
            print(f"  Embeddings criados: {all_embeddings.shape}")

            return model, all_embeddings

        except Exception as e:
            print(f"  Erro ao criar embeddings: {e}")
            return None, None

    def create_faiss_index(self, embeddings):
        """Cria índice Faiss para busca rápida."""
        print("\nConstruindo índice Faiss...")
        try:
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            print(f"  Índice criado: {index.ntotal} vetores, dimensão {dimension}")
            return index
        except Exception as e:
            print(f"  Erro ao criar índice: {e}")
            return None

    def save_system(self, model, index, chunks, metadata, documents):
        """Salva todo o sistema RAG."""
        print(f"\nSalvando sistema em: {os.path.abspath(self.save_dir)}")

        try:
            os.makedirs(self.save_dir, exist_ok=True)

            # Salva modelo de embedding
            print("  Salvando modelo de embedding...")
            model.save(f"{self.save_dir}/embedding_model")

            # Salva índice Faiss
            print("  Salvando índice Faiss...")
            faiss.write_index(index, f"{self.save_dir}/faiss_index.bin")

            # Salva dados - CORRIGIDO: conversão de tipos numpy
            print("  Salvando dados...")

            # Converter metadata para tipos nativos
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
                'metadata': metadata_clean,
                'documents': documents,
                'stats': {
                    'num_chunks': len(chunks),
                    'num_documents': len(documents),
                    'embedding_dim': int(index.d),  # Converter para int nativo
                    'domains': list(set(doc['domain'] for doc in documents))
                }
            }

            with open(f"{self.save_dir}/rag_data.pkl", "wb") as f:
                pickle.dump(data, f)

            # Salva informações do sistema
            info = {
                'created_at': '2024-01-01',
                'model_name': 'all-MiniLM-L6-v2',
                'num_chunks': len(chunks),
                'num_documents': len(documents),
                'embedding_dimension': int(index.d),  # Converter para int nativo
                'domains': [doc['domain'] for doc in documents],
                'urls': [doc['url'] for doc in documents],
                'system_ready': True,
                'type': 'universal'
            }

            with open(f"{self.save_dir}/system_info.json", "w") as f:
                json.dump(info, f, indent=2)

            print("  Sistema salvo com sucesso!")

            # Mostra estatísticas
            print("\nESTATÍSTICAS:")
            domains = [doc['domain'] for doc in documents]
            unique_domains = set(domains)
            print(f"  Domínios processados: {len(unique_domains)}")
            for domain in unique_domains:
                count = domains.count(domain)
                print(f"    - {domain}: {count} página(s)")

            return True

        except Exception as e:
            print(f"  Erro ao salvar: {e}")
            return False

    def train(self):
        """Executa o treinamento completo."""
        print("INICIANDO TREINAMENTO RAG UNIVERSAL")
        print("=" * 60)
        print("Otimizado e corrigido")
        print("Funciona com qualquer site!")
        print("=" * 60)

        # 1. Coleta documentos
        print("\nCOLETANDO DOCUMENTOS...")
        documents = []
        failed_urls = []

        for i, url in enumerate(self.articles, 1):
            print(f"\n[{i}/{len(self.articles)}] Processando...")
            doc = self.get_article_content_universal(url)
            if doc:
                documents.append(doc)
            else:
                failed_urls.append(url)

            # Pausa entre requisições para ser respeitoso
            if i < len(self.articles):
                time.sleep(1)

        if not documents:
            print("Nenhum documento coletado!")
            if failed_urls:
                print("\nURLs que falharam:")
                for url in failed_urls:
                    print(f"  - {url}")
            return False

        print(f"\n{len(documents)} documentos coletados com sucesso")
        if failed_urls:
            print(f"{len(failed_urls)} URLs falharam")

        # 2. Cria chunks
        chunks, metadata = self.create_chunks(documents)
        if not chunks:
            print("Nenhum chunk criado!")
            return False

        # 3. Cria embeddings
        model, embeddings = self.create_embeddings(chunks)
        if model is None or embeddings is None:
            print("Falha ao criar embeddings!")
            return False

        # 4. Cria índice
        index = self.create_faiss_index(embeddings)
        if index is None:
            print("Falha ao criar índice!")
            return False

        # 5. Salva sistema
        if self.save_system(model, index, chunks, metadata, documents):
            print("\nTREINAMENTO UNIVERSAL CONCLUÍDO!")
            print("=" * 60)
            print("Sistema RAG pronto para qualquer conteúdo")
            print("Execute agora: python 2_rag_query_smart.py")
            print("Ou execute: python 3_rag_feedback_super_robust_corrigido.py")
            return True
        else:
            print("Falha ao salvar sistema!")
            return False

def main():
    """Função principal."""
    print("RAG TRAINER UNIVERSAL")
    print("Funciona com Wikipedia, blogs, notícias, documentação, etc.\n")

    try:
        trainer = UniversalRAGTrainer()

        print("URLs configuradas:")
        for i, url in enumerate(trainer.articles, 1):
            domain = urlparse(url).netloc
            print(f"  {i}. {domain}")

        print("\nPara usar outros sites, edite a lista 'self.articles' no código")
        print("Pressione Ctrl+C para cancelar\n")

        success = trainer.train()

        if success:
            print("\nPronto! Agora você pode usar os outros programas.")
        else:
            print("\nTreinamento falhou. Verifique as URLs e sua conexão.")

    except KeyboardInterrupt:
        print("\n\nTreinamento cancelado pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
