#!/usr/bin/env python3
"""
PROGRAMA 2: CONSULTA RAG INTELIGENTE
Usa o aprendizado do programa 3 para dar respostas melhores
Otimizado para MacBook
"""

import pickle
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import sys

class SmartRAGQuery:
    def __init__(self, save_dir="./rag_system_mac"):
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.generator = None

        # Sistema de memória (carregado do programa 3)
        self.document_scores = {}
        self.has_memory = False
        self.total_feedback_queries = 0

    def load_system(self):
        """Carrega o sistema RAG treinado."""
        print("🔄 Carregando sistema RAG inteligente...")
        print(f"📁 Diretório: {os.path.abspath(self.save_dir)}")

        if not os.path.exists(self.save_dir):
            print(f"❌ Sistema não encontrado em {self.save_dir}")
            print("Execute primeiro: python 1_rag_trainer.py")
            return False

        try:
            # Carrega componentes básicos
            print("  📐 Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            print("  🔍 Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            print("  📊 Carregando dados...")
            with open(f"{self.save_dir}/rag_data.pkl", "rb") as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.metadata = data['metadata']

            print("  🤖 Carregando gerador de texto...")
            self.generator = pipeline(
                "text2text-generation",
                model="t5-small",
                max_length=200,
                do_sample=False
            )

            # TENTA CARREGAR MEMÓRIA DO PROGRAMA 3
            self.load_feedback_memory()

            print("✅ Sistema carregado com sucesso!")
            print(f"📊 {len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def load_feedback_memory(self):
        """Carrega memória de feedback do programa 3."""
        try:
            memory_file = f"{self.save_dir}/persistent_memory.json"

            if os.path.exists(memory_file):
                print("  🧠 Carregando memória de feedback...")
                with open(memory_file, "r", encoding='utf-8') as f:
                    memory_data = json.load(f)

                    self.document_scores = memory_data.get("document_scores", {})
                    query_history = memory_data.get("query_history", [])
                    self.total_feedback_queries = len(query_history)

                    if self.document_scores:
                        self.has_memory = True
                        docs_with_feedback = len(self.document_scores)

                        print(f"  🎉 Memória encontrada!")
                        print(f"     📈 {self.total_feedback_queries} consultas com feedback")
                        print(f"     📚 {docs_with_feedback} documentos treinados")

                        # Mostra alguns documentos bem avaliados
                        best_docs = self.get_best_documents(3)
                        if best_docs:
                            print(f"     🏆 Melhores documentos:")
                            for doc_id, avg_score in best_docs:
                                title = self.metadata[int(doc_id)]['title']
                                print(f"        - {title}: {avg_score:.1f}/5")

                        print("  ✅ Sistema agora usa inteligência do programa 3!")
                    else:
                        print("  📊 Memória vazia - usando busca básica")
            else:
                print("  📊 Nenhuma memória encontrada - usando busca básica")
                print("  💡 Execute o programa 3 para treinar o sistema!")

        except Exception as e:
            print(f"  ⚠️ Erro ao carregar memória: {e}")
            print("  🔄 Usando busca básica")

    def get_best_documents(self, n=5):
        """Retorna os N melhores documentos por avaliação."""
        doc_averages = []
        for doc_id, scores in self.document_scores.items():
            if len(scores) >= 1:
                avg_score = sum(scores) / len(scores)
                doc_averages.append((doc_id, avg_score))

        doc_averages.sort(key=lambda x: x[1], reverse=True)
        return doc_averages[:n]

    def calculate_learned_weight(self, chunk_id):
        """Calcula peso baseado no aprendizado do programa 3."""
        if not self.has_memory:
            return 1.0  # Peso neutro se não há memória

        chunk_str = str(chunk_id)

        if chunk_str in self.document_scores:
            scores = self.document_scores[chunk_str]
            if len(scores) >= 1:
                # Calcula média das avaliações
                avg_score = sum(scores) / len(scores)

                # Converte para peso (1-5 -> 0.3-2.2)
                if avg_score >= 4.5:
                    return 2.2  # Documentos excelentes
                elif avg_score >= 4.0:
                    return 1.9  # Documentos muito bons
                elif avg_score >= 3.5:
                    return 1.5  # Documentos bons
                elif avg_score >= 2.5:
                    return 1.0  # Documentos neutros
                elif avg_score >= 1.5:
                    return 0.7  # Documentos ruins
                else:
                    return 0.3  # Documentos muito ruins

        return 1.0  # Peso neutro para documentos sem feedback

    def smart_search(self, question, k=5):
        """Busca inteligente usando aprendizado do programa 3."""
        try:
            # Cria embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Busca mais documentos para ter opções
            search_k = k * 2 if self.has_memory else k
            scores, indices = self.index.search(question_embedding, search_k)

            # Aplica pesos aprendidos (se houver memória)
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    base_score = float(scores[0][i])

                    if self.has_memory:
                        learned_weight = self.calculate_learned_weight(idx)
                        adjusted_score = base_score / learned_weight  # Menor = melhor

                        # Determina confiança baseada no aprendizado
                        if learned_weight > 1.8:
                            confidence = "🔥"  # Muito confiável
                        elif learned_weight > 1.2:
                            confidence = "👍"  # Confiável
                        elif learned_weight < 0.8:
                            confidence = "⚠️"  # Pouco confiável
                        else:
                            confidence = "📊"  # Neutro
                    else:
                        learned_weight = 1.0
                        adjusted_score = base_score
                        confidence = "📊"  # Neutro (sem memória)

                    results.append({
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'base_score': base_score,
                        'learned_weight': learned_weight,
                        'adjusted_score': adjusted_score,
                        'confidence': confidence,
                        'rank': i + 1,
                        'chunk_id': idx
                    })

            # Ordena por score ajustado se há memória, senão por score base
            if self.has_memory:
                results.sort(key=lambda x: x['adjusted_score'])
            else:
                results.sort(key=lambda x: x['base_score'])

            return results[:k]

        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []

    def generate_smart_answer(self, question, context_docs):
        """Gera resposta priorizando documentos bem avaliados."""
        try:
            # Se há memória, ordena por peso aprendido
            if self.has_memory:
                sorted_docs = sorted(context_docs, key=lambda x: x['learned_weight'], reverse=True)
            else:
                sorted_docs = context_docs

            # Cria contexto com os melhores documentos
            context_parts = []
            for doc in sorted_docs[:3]:  # Usa os 3 melhores
                title = doc['metadata']['title']
                text = doc['text']
                confidence = doc['confidence']
                context_parts.append(f"Fonte {confidence} ({title}): {text}")

            context = "\n\n".join(context_parts)

            # Limita tamanho do contexto
            if len(context) > 1000:
                context = context[:1000] + "..."

            # Cria prompt
            if self.has_memory:
                prompt_prefix = "Com base no contexto verificado e otimizado"
            else:
                prompt_prefix = "Com base no contexto fornecido"

            prompt = f"""{prompt_prefix}, responda a pergunta de forma clara e precisa.

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
        """Executa uma consulta inteligente."""
        print("\n" + "="*60)
        if self.has_memory:
            print("🔍 PROCESSANDO COM INTELIGÊNCIA APRENDIDA")
        else:
            print("🔍 PROCESSANDO SUA PERGUNTA")
        print("="*60)
        print(f"❓ Pergunta: {question}")

        # Busca documentos (inteligente se há memória)
        if self.has_memory:
            print("\n🧠 Buscando com inteligência do programa 3...")
        else:
            print("\n📚 Buscando documentos relevantes...")

        docs = self.smart_search(question, k=5)

        if not docs:
            print("❌ Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes para sua pergunta."

        # Mostra documentos encontrados
        print(f"\n📋 DOCUMENTOS SELECIONADOS:")
        for doc in docs[:3]:
            if self.has_memory:
                print(f"  {doc['rank']}. {doc['metadata']['title']} {doc['confidence']}")
                print(f"     Peso aprendido: {doc['learned_weight']:.2f}")
                print(f"     Score: {doc['base_score']:.3f} → {doc['adjusted_score']:.3f}")
            else:
                print(f"  {doc['rank']}. {doc['metadata']['title']} {doc['confidence']}")
                print(f"     Score: {doc['base_score']:.3f}")
            print(f"     Texto: {doc['text'][:80]}...")
            print()

        # Gera resposta
        if self.has_memory:
            print("🤖 Gerando resposta otimizada...")
        else:
            print("🤖 Gerando resposta...")

        answer = self.generate_smart_answer(question, docs)

        return answer

    def interactive_mode(self):
        """Modo interativo inteligente."""
        if self.has_memory:
            print("🚀 CONSULTA RAG INTELIGENTE")
            print("="*50)
            print("🧠 Sistema usando aprendizado do programa 3!")
            print(f"📊 Baseado em {self.total_feedback_queries} consultas com feedback")
            print("💡 Respostas otimizadas com base nas suas avaliações")
        else:
            print("🚀 CONSULTA RAG BÁSICA")
            print("="*50)
            print("📊 Sistema usando busca por similaridade")
            print("💡 Execute o programa 3 para treinar o sistema!")

        print("⌨️  Digite 'sair' para terminar")
        print("="*50)

        while True:
            try:
                print("\n" + "🎯" + "="*58)
                question = input("❓ Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    if self.has_memory:
                        print("\n🧠 Obrigado por usar o sistema inteligente!")
                        print("💡 Continue treinando com o programa 3 para melhorar ainda mais!")
                    else:
                        print("\n👋 Até logo!")
                        print("💡 Execute o programa 3 para treinar o sistema!")
                    break

                if len(question) < 3:
                    print("⚠️  Pergunta muito curta. Tente novamente.")
                    continue

                # Processa pergunta
                answer = self.query(question)

                # Mostra resposta
                if self.has_memory:
                    print("\n🤖 RESPOSTA INTELIGENTE:")
                else:
                    print("\n🤖 RESPOSTA:")
                print("-" * 60)
                print(answer)
                print("-" * 60)

                if not self.has_memory:
                    print("\n💡 Dica: Execute o programa 3 para treinar o sistema!")
                    print("   Suas avaliações tornarão as respostas muito melhores!")

            except KeyboardInterrupt:
                print("\n\n⏹️  Programa interrompido.")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

def main():
    """Função principal."""
    print("🍎 RAG QUERY INTELIGENTE PARA MACBOOK")
    print("Usa o aprendizado do programa 3 automaticamente\n")

    try:
        # Inicializa sistema
        rag = SmartRAGQuery()

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
