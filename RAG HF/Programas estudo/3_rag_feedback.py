#!/usr/bin/env python3
"""
PROGRAMA 3: RAG COM FEEDBACK
Sistema RAG que aprende com avaliações do usuário (1-5)
Otimizado para MacBook
"""

import pickle
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from datetime import datetime
import sys

class RAGWithFeedback:
    def __init__(self, save_dir="./rag_system_mac"):
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.generator = None
        self.feedback_data = {}
        self.document_scores = {}
        self.query_history = []

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

            # Carrega dados de feedback anteriores
            self.load_feedback_data()

            print("✅ Sistema carregado com sucesso!")
            print(f"📊 {len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def load_feedback_data(self):
        """Carrega histórico de feedback."""
        try:
            feedback_file = f"{self.save_dir}/feedback_history.json"
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    self.feedback_data = data.get("feedback_data", {})
                    self.document_scores = data.get("document_scores", {})
                    self.query_history = data.get("query_history", [])
                print(f"  📊 Carregado histórico: {len(self.query_history)} consultas anteriores")
            else:
                print("  📊 Nenhum histórico encontrado - começando do zero")
        except Exception as e:
            print(f"  ⚠️ Erro ao carregar histórico: {e}")

    def save_feedback_data(self):
        """Salva dados de feedback."""
        try:
            feedback_file = f"{self.save_dir}/feedback_history.json"
            data = {
                "feedback_data": self.feedback_data,
                "document_scores": self.document_scores,
                "query_history": self.query_history,
                "last_updated": datetime.now().isoformat(),
                "total_queries": len(self.query_history)
            }
            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar feedback: {e}")

    def calculate_document_weight(self, chunk_id):
        """Calcula peso do documento baseado no feedback."""
        chunk_str = str(chunk_id)
        if chunk_str in self.document_scores:
            scores = self.document_scores[chunk_str]
            if scores:
                avg_score = sum(scores) / len(scores)
                # Converte nota 1-5 para peso 0.3-2.0
                weight = 0.3 + (avg_score - 1) * 0.425  # (5-1) * 0.425 = 1.7, então 0.3 + 1.7 = 2.0
                return max(0.3, min(2.0, weight))
        return 1.0  # Peso neutro para documentos sem feedback

    def search_with_learning(self, question, k=5):
        """Busca documentos considerando feedback anterior."""
        try:
            # Cria embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Busca mais documentos para ter opções
            scores, indices = self.index.search(question_embedding, k*2)

            # Aplica pesos baseados no feedback
            weighted_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    base_score = float(scores[0][i])
                    weight = self.calculate_document_weight(idx)
                    adjusted_score = base_score / weight  # Menor score = melhor

                    weighted_results.append({
                        'chunk_id': idx,
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'base_score': base_score,
                        'weight': weight,
                        'adjusted_score': adjusted_score,
                        'rank': i + 1
                    })

            # Ordena por score ajustado e pega os melhores
            weighted_results.sort(key=lambda x: x['adjusted_score'])
            return weighted_results[:k]

        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []

    def generate_smart_answer(self, question, context_docs):
        """Gera resposta priorizando documentos bem avaliados."""
        try:
            # Ordena documentos por peso (melhores primeiro)
            sorted_docs = sorted(context_docs, key=lambda x: x['weight'], reverse=True)

            # Cria contexto priorizando documentos bem avaliados
            context_parts = []
            for doc in sorted_docs[:3]:  # Usa os 3 melhores
                title = doc['metadata']['title']
                text = doc['text']
                context_parts.append(f"Fonte ({title}): {text}")

            context = "\n\n".join(context_parts)

            # Limita tamanho do contexto
            if len(context) > 1000:
                context = context[:1000] + "..."

            # Cria prompt otimizado
            prompt = f"""Baseado no contexto fornecido, responda a pergunta de forma clara e precisa.

Pergunta: {question}

Contexto:
{context}

Resposta detalhada:"""

            # Gera resposta
            response = self.generator(prompt, max_length=150, num_return_sequences=1)

            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Limpa a resposta
                if "Resposta detalhada:" in answer:
                    answer = answer.split("Resposta detalhada:")[-1].strip()
                elif "Resposta:" in answer:
                    answer = answer.split("Resposta:")[-1].strip()
                return answer
            else:
                return "Não consegui gerar uma resposta adequada."

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return f"Erro: {str(e)}"

    def record_feedback(self, question, answer, docs_used, rating):
        """Registra feedback do usuário."""
        try:
            # Cria registro da consulta
            query_record = {
                "id": f"query_{len(self.query_history)}",
                "question": question,
                "answer": answer,
                "rating": rating,
                "timestamp": datetime.now().isoformat(),
                "docs_used": [doc['chunk_id'] for doc in docs_used]
            }
            self.query_history.append(query_record)

            # Atualiza scores dos documentos
            for doc in docs_used:
                chunk_id = str(doc['chunk_id'])
                if chunk_id not in self.document_scores:
                    self.document_scores[chunk_id] = []
                self.document_scores[chunk_id].append(rating)

                # Mantém apenas os últimos 15 feedbacks por documento
                if len(self.document_scores[chunk_id]) > 15:
                    self.document_scores[chunk_id] = self.document_scores[chunk_id][-15:]

            # Salva dados
            self.save_feedback_data()

            # Mostra estatísticas
            self.show_learning_stats(rating)

        except Exception as e:
            print(f"⚠️ Erro ao registrar feedback: {e}")

    def show_learning_stats(self, current_rating):
        """Mostra estatísticas de aprendizado."""
        if len(self.query_history) >= 1:
            print(f"\n📊 ESTATÍSTICAS DE APRENDIZADO:")
            print(f"   📝 Consulta atual: #{len(self.query_history)}")
            print(f"   ⭐ Nota atual: {current_rating}/5")

            if len(self.query_history) > 1:
                recent_ratings = [q['rating'] for q in self.query_history[-5:]]
                avg_recent = sum(recent_ratings) / len(recent_ratings)
                print(f"   📈 Média últimas 5: {avg_recent:.1f}/5")

            print(f"   📚 Documentos com feedback: {len(self.document_scores)}")

            # Feedback motivacional
            if current_rating >= 4:
                print("   🎉 Excelente! O sistema está aprendendo com você.")
            elif current_rating >= 3:
                print("   👍 Boa resposta. Continuando a melhorar.")
            else:
                print("   🔄 Resposta ruim. Ajustando para próximas consultas.")

    def query_with_feedback(self, question):
        """Executa consulta completa com sistema de feedback."""
        print("\n" + "="*60)
        print("🔍 PROCESSANDO SUA PERGUNTA")
        print("="*60)
        print(f"❓ Pergunta: {question}")

        # Busca documentos com aprendizado
        print("\n📚 Buscando documentos (considerando feedback anterior)...")
        docs = self.search_with_learning(question, k=5)

        if not docs:
            print("❌ Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes.", []

        # Mostra documentos encontrados
        print(f"\n📋 DOCUMENTOS SELECIONADOS:")
        for i, doc in enumerate(docs[:3], 1):
            weight_emoji = "🔥" if doc['weight'] > 1.3 else "👍" if doc['weight'] > 0.7 else "⚠️"
            print(f"  {i}. {doc['metadata']['title']} {weight_emoji}")
            print(f"     Peso: {doc['weight']:.2f} | Score: {doc['base_score']:.3f}")
            print(f"     Texto: {doc['text'][:80]}...")
            print()

        # Gera resposta inteligente
        print("🤖 Gerando resposta otimizada...")
        answer = self.generate_smart_answer(question, docs)

        return answer, docs[:3]  # Retorna apenas os 3 melhores

    def interactive_mode(self):
        """Modo interativo com sistema de feedback."""
        print("🚀 RAG INTELIGENTE COM FEEDBACK")
        print("="*60)
        print("🧠 Este sistema aprende com suas avaliações!")
        print("📊 Notas: 1-2 = Ruim | 3 = OK | 4-5 = Excelente")
        print("⌨️  Digite 'sair' para terminar")
        print("="*60)

        session_queries = 0

        while True:
            try:
                print("\n" + "🎯" + "="*58)
                question = input("❓ Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    self.show_session_summary(session_queries)
                    break

                if len(question) < 3:
                    print("⚠️  Pergunta muito curta. Tente novamente.")
                    continue

                # Processa pergunta
                answer, docs_used = self.query_with_feedback(question)

                # Mostra resposta
                print("\n🤖 RESPOSTA:")
                print("-" * 60)
                print(answer)
                print("-" * 60)

                # Solicita avaliação
                print("\n⭐ AVALIE A RESPOSTA:")
                print("1 = Muito ruim | 2 = Ruim | 3 = OK | 4 = Boa | 5 = Excelente")

                while True:
                    try:
                        rating_input = input("Sua nota (1-5): ").strip()
                        rating = int(rating_input)
                        if 1 <= rating <= 5:
                            break
                        else:
                            print("Por favor, digite um número entre 1 e 5.")
                    except ValueError:
                        print("Por favor, digite um número válido.")

                # Registra feedback
                self.record_feedback(question, answer, docs_used, rating)
                session_queries += 1

                # Resposta motivacional
                if rating < 3:
                    print("\n🔄 Obrigado! Vou melhorar nas próximas respostas.")
                elif rating == 3:
                    print("\n👍 Obrigado! Continuarei aprendendo.")
                else:
                    print("\n🎉 Ótimo! Estou aprendendo com seu feedback positivo.")

            except KeyboardInterrupt:
                print("\n\n⏹️  Programa interrompido.")
                self.show_session_summary(session_queries)
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

    def show_session_summary(self, session_queries):
        """Mostra resumo da sessão."""
        print("\n📊 RESUMO DA SESSÃO:")
        print("="*40)
        print(f"🔢 Consultas nesta sessão: {session_queries}")
        print(f"📈 Total de consultas: {len(self.query_history)}")

        if self.query_history:
            recent_ratings = [q['rating'] for q in self.query_history[-session_queries:]] if session_queries > 0 else []
            if recent_ratings:
                avg_session = sum(recent_ratings) / len(recent_ratings)
                print(f"⭐ Nota média da sessão: {avg_session:.1f}/5")

        print("\n👋 Obrigado por ajudar o sistema a aprender!")

def main():
    """Função principal."""
    print("🍎 RAG COM FEEDBACK PARA MACBOOK")
    print("Sistema que aprende com suas avaliações\n")

    try:
        # Inicializa sistema
        rag = RAGWithFeedback()

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
