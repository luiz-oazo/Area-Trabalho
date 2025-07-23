#!/usr/bin/env python3
"""
PROGRAMA 2: CONSULTA RAG
Faz perguntas e recebe respostas do sistema RAG
Otimizado para MacBook
"""

import pickle
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys

class RAGQuery:
    def __init__(self, save_dir="./rag_system_mac"):
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.generator = None

    def load_system(self):
        """Carrega o sistema RAG treinado."""
        print("🔄 Carregando sistema RAG...")
        print(f"📁 Diretório: {os.path.abspath(self.save_dir)}")

        if not os.path.exists(self.save_dir):
            print(f"❌ Sistema não encontrado em {self.save_dir}")
            print("Execute primeiro: python 1_rag_trainer.py")
            return False

        try:
            # Carrega modelo de embedding
            print("  📐 Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            # Carrega índice Faiss
            print("  🔍 Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            # Carrega dados
            print("  📊 Carregando dados...")
            with open(f"{self.save_dir}/rag_data.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']

            # Carrega gerador de texto
            print("  🤖 Carregando gerador de texto...")
            self.generator = pipeline(
                "text2text-generation",
                model="t5-small",
                max_length=200,
                do_sample=False
            )

            print("✅ Sistema carregado com sucesso!")
            print(f"📊 {len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def search_documents(self, question, k=5):
        """Busca documentos relevantes para a pergunta."""
        try:
            # Cria embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Busca no índice
            scores, indices = self.index.search(question_embedding, k)

            # Recupera documentos
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    results.append({
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'score': float(scores[0][i]),
                        'rank': i + 1
                    })

            return results

        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []

    def generate_answer(self, question, context_docs):
        """Gera resposta baseada no contexto encontrado."""
        try:
            # Cria contexto com os melhores documentos
            context_parts = []
            for doc in context_docs[:3]:  # Usa apenas os 3 melhores
                title = doc['metadata']['title']
                text = doc['text']
                context_parts.append(f"Fonte ({title}): {text}")

            context = "\n\n".join(context_parts)

            # Limita tamanho do contexto
            if len(context) > 1000:
                context = context[:1000] + "..."

            # Cria prompt
            prompt = f"""Responda a pergunta baseado no contexto fornecido.

Pergunta: {question}

Contexto:
{context}

Resposta:"""

            # Gera resposta
            response = self.generator(prompt, max_length=150, num_return_sequences=1)

            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Limpa a resposta
                if "Resposta:" in answer:
                    answer = answer.split("Resposta:")[-1].strip()
                return answer
            else:
                return "Não consegui gerar uma resposta adequada."

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return f"Erro: {str(e)}"

    def query(self, question):
        """Executa uma consulta completa."""
        print("\n" + "="*60)
        print("🔍 PROCESSANDO SUA PERGUNTA")
        print("="*60)
        print(f"❓ Pergunta: {question}")

        # Busca documentos relevantes
        print("\n📚 Buscando documentos relevantes...")
        docs = self.search_documents(question, k=5)

        if not docs:
            print("❌ Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes para sua pergunta."

        # Mostra documentos encontrados
        print(f"\n📋 DOCUMENTOS ENCONTRADOS:")
        for doc in docs[:3]:
            print(f"  {doc['rank']}. {doc['metadata']['title']}")
            print(f"     Score: {doc['score']:.3f}")
            print(f"     Texto: {doc['text'][:80]}...")
            print()

        # Gera resposta
        print("🤖 Gerando resposta...")
        answer = self.generate_answer(question, docs)

        return answer

    def interactive_mode(self):
        """Modo interativo para fazer perguntas."""
        print("🚀 MODO CONSULTA RAG")
        print("="*50)
        print("💡 Digite suas perguntas sobre Inteligência Artificial")
        print("⌨️  Digite 'sair' para terminar")
        print("="*50)

        while True:
            try:
                print("\n" + "🎯" + "="*58)
                question = input("❓ Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    print("\n👋 Até logo!")
                    break

                if len(question) < 3:
                    print("⚠️  Pergunta muito curta. Tente novamente.")
                    continue

                # Processa pergunta
                answer = self.query(question)

                # Mostra resposta
                print("\n🤖 RESPOSTA:")
                print("-" * 60)
                print(answer)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\n⏹️  Programa interrompido.")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

def main():
    """Função principal."""
    print("🍎 RAG QUERY PARA MACBOOK")
    print("Consulte seu sistema de IA treinado\n")

    try:
        # Inicializa sistema
        rag = RAGQuery()

        # Carrega sistema
        if not rag.load_system():
            print("\n💡 Execute primeiro o treinamento:")
            print("   python 1_rag_trainer.py")
            sys.exit(1)

        # Inicia modo interativo
        rag.interactive_mode()

    except KeyboardInterrupt:
        print("\n\n⏹️  Programa cancelado.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
