import pickle
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from datetime import datetime
import random

class SmartRAGWithFeedback:
    def __init__(self, save_dir="/content/simple_rag_system"):  # CORRIGIDO: mesmo diretório do treinador
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.dataset = None
        self.generator = None
        self.tokenizer = None
        self.feedback_data = {}
        self.document_scores = {}
        self.query_history = []

    def load_system(self):
        """Carrega o sistema RAG com dados de feedback."""
        try:
            print("🔄 Carregando sistema RAG inteligente...")

            if not os.path.exists(self.save_dir):
                print(f"❌ Diretório {self.save_dir} não encontrado!")
                print("Execute primeiro o simple_rag_trainer.py para criar o sistema.")
                return False

            # Carregar componentes básicos
            print("  Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            print("  Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            print("  Carregando dataset...")
            with open(f"{self.save_dir}/dataset.pkl", "rb") as f:
                self.dataset = pickle.load(f)

            # Carregar modelo gerador
            print("  Carregando modelo gerador...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
                model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
                self.generator = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                    max_length=200,
                    do_sample=False
                )
            except Exception as e:
                print(f"    Usando pipeline padrão: {e}")
                self.generator = pipeline("text2text-generation", model="t5-small")

            # Carregar dados de feedback
            self.load_feedback_data()

            print("✅ Sistema carregado com sucesso!")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def load_feedback_data(self):
        """Carrega dados de feedback anteriores."""
        try:
            feedback_file = f"{self.save_dir}/feedback_data.json"
            if os.path.exists(feedback_file):
                with open(feedback_file, "r") as f:
                    data = json.load(f)
                    self.feedback_data = data.get("feedback_data", {})
                    self.document_scores = data.get("document_scores", {})
                    self.query_history = data.get("query_history", [])
                print(f"  📊 Carregados {len(self.feedback_data)} feedbacks anteriores")
            else:
                print("  📊 Nenhum feedback anterior encontrado - começando do zero")
        except Exception as e:
            print(f"  ⚠️ Erro ao carregar feedback: {e}")

    def save_feedback_data(self):
        """Salva dados de feedback."""
        try:
            feedback_file = f"{self.save_dir}/feedback_data.json"
            data = {
                "feedback_data": self.feedback_data,
                "document_scores": self.document_scores,
                "query_history": self.query_history,
                "last_updated": datetime.now().isoformat()
            }
            with open(feedback_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Erro ao salvar feedback: {e}")

    def calculate_document_relevance_score(self, doc_id):
        """Calcula pontuação de relevância de um documento baseada no feedback."""
        if str(doc_id) in self.document_scores:
            scores = self.document_scores[str(doc_id)]
            if scores:
                avg_score = sum(scores) / len(scores)
                # Normalizar para peso entre 0.5 e 2.0
                weight = 0.5 + (avg_score - 1) * 0.375  # (5-1) * 0.375 = 1.5, então 0.5 + 1.5 = 2.0
                return max(0.5, min(2.0, weight))
        return 1.0  # Peso neutro para documentos sem feedback

    def retrieve_documents_with_learning(self, question, k=5):
        """Recupera documentos considerando feedback anterior."""
        try:
            # Buscar mais documentos inicialmente
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Buscar top-k*2 para ter mais opções
            scores, indices = self.index.search(question_embedding, k*2)

            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.dataset['chunks']) and idx >= 0:
                    chunk = self.dataset['chunks'][idx]
                    metadata = self.dataset['metadata'][idx]

                    # Calcular pontuação ajustada pelo feedback
                    base_score = float(scores[0][i])
                    relevance_weight = self.calculate_document_relevance_score(idx)
                    adjusted_score = base_score / relevance_weight  # Menor score = melhor

                    doc = {
                        'title': metadata['title'],
                        'text': chunk,
                        'base_score': base_score,
                        'relevance_weight': relevance_weight,
                        'adjusted_score': adjusted_score,
                        'url': metadata['url'],
                        'chunk_id': idx,
                        'doc_id': idx
                    }
                    retrieved_docs.append(doc)

            # Ordenar por pontuação ajustada e pegar os top-k
            retrieved_docs.sort(key=lambda x: x['adjusted_score'])
            return retrieved_docs[:k]

        except Exception as e:
            print(f"❌ Erro ao recuperar documentos: {e}")
            return []

    def generate_answer_with_context(self, question, context_docs):
        """Gera resposta melhorada baseada no contexto."""
        try:
            # Priorizar documentos com melhor feedback
            sorted_docs = sorted(context_docs, key=lambda x: x['relevance_weight'], reverse=True)

            # Criar contexto priorizando documentos bem avaliados
            context_parts = []
            for doc in sorted_docs[:3]:  # Usar apenas os 3 melhores
                context_parts.append(f"Fonte ({doc['title']}): {doc['text']}")

            context = "\n\n".join(context_parts)

            # Limitar tamanho do contexto
            max_context_length = 900
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."

            # Criar prompt melhorado
            prompt = f"""Baseado no contexto fornecido, responda a pergunta de forma clara e precisa.

Pergunta: {question}

Contexto:
{context}

Resposta detalhada:"""

            # Gerar resposta
            response = self.generator(prompt, max_length=150, num_return_sequences=1)

            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Limpar resposta
                if "Resposta detalhada:" in answer:
                    answer = answer.split("Resposta detalhada:")[-1].strip()
                elif "Resposta:" in answer:
                    answer = answer.split("Resposta:")[-1].strip()
                return answer
            else:
                return "Não foi possível gerar uma resposta adequada."

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return f"Erro ao processar: {str(e)}"

    def record_feedback(self, question, answer, docs_used, rating):
        """Registra feedback do usuário."""
        try:
            # Criar ID único para esta consulta
            query_id = f"query_{len(self.query_history)}"

            # Registrar na história
            query_record = {
                "id": query_id,
                "question": question,
                "answer": answer,
                "rating": rating,
                "timestamp": datetime.now().isoformat(),
                "docs_used": [doc['doc_id'] for doc in docs_used]
            }
            self.query_history.append(query_record)

            # Atualizar pontuações dos documentos
            for doc in docs_used:
                doc_id = str(doc['doc_id'])
                if doc_id not in self.document_scores:
                    self.document_scores[doc_id] = []
                self.document_scores[doc_id].append(rating)

                # Manter apenas os últimos 10 feedbacks por documento
                if len(self.document_scores[doc_id]) > 10:
                    self.document_scores[doc_id] = self.document_scores[doc_id][-10:]

            # Salvar dados
            self.save_feedback_data()

            # Mostrar estatísticas
            self.show_learning_stats(rating)

        except Exception as e:
            print(f"⚠️ Erro ao registrar feedback: {e}")

    def show_learning_stats(self, current_rating):
        """Mostra estatísticas de aprendizado."""
        if len(self.query_history) > 1:
            recent_ratings = [q['rating'] for q in self.query_history[-5:]]
            avg_recent = sum(recent_ratings) / len(recent_ratings)

            print(f"\n📊 ESTATÍSTICAS DE APRENDIZADO:")
            print(f"   Nota atual: {current_rating}/5")
            print(f"   Média das últimas 5 consultas: {avg_recent:.1f}/5")
            print(f"   Total de consultas: {len(self.query_history)}")
            print(f"   Documentos com feedback: {len(self.document_scores)}")

            if current_rating >= 4:
                print("   🎉 Ótima resposta! O sistema está aprendendo.")
            elif current_rating >= 3:
                print("   👍 Resposta boa. Continuando a melhorar.")
            else:
                print("   🔄 Resposta ruim. Ajustando para próximas consultas.")

    def query_with_feedback(self, question):
        """Executa consulta completa com sistema de feedback."""
        print("\n" + "="*60)
        print("🔍 PROCESSANDO SUA PERGUNTA...")
        print("="*60)

        # Recuperar documentos com aprendizado
        print("📚 Buscando documentos relevantes (considerando feedback anterior)...")
        docs = self.retrieve_documents_with_learning(question, k=3)

        if not docs:
            return "❌ Nenhum documento relevante encontrado.", []

        # Mostrar documentos encontrados
        print(f"\n📋 DOCUMENTOS SELECIONADOS:")
        for i, doc in enumerate(docs, 1):
            weight_status = "🔥" if doc['relevance_weight'] > 1.2 else "👍" if doc['relevance_weight'] > 0.8 else "⚠️"
            print(f"   {i}. {doc['title']} {weight_status}")
            print(f"      Peso de relevância: {doc['relevance_weight']:.2f}")
            print(f"      Texto: {doc['text'][:100]}...")
            print()

        # Gerar resposta
        print("🤖 Gerando resposta otimizada...")
        answer = self.generate_answer_with_context(question, docs)

        return answer, docs

def main():
    """Função principal com sistema de feedback."""
    print("🚀 SISTEMA RAG INTELIGENTE COM FEEDBACK")
    print("="*60)
    print("Este sistema aprende com suas avaliações!")
    print("Notas 1-2: Resposta ruim | 3: OK | 4-5: Muito boa")
    print("="*60)

    # Inicializar sistema
    rag = SmartRAGWithFeedback()

    if not rag.load_system():
        print("❌ Execute primeiro o simple_rag_trainer.py")
        return

    print("\n✅ Sistema carregado e pronto!")

    while True:
        print("\n" + "🎯" + "="*58)
        question = input("❓ Sua pergunta (ou 'sair' para terminar): ").strip()

        if question.lower() in ['sair', 'exit', 'quit', '']:
            print("\n📊 RESUMO DA SESSÃO:")
            if rag.query_history:
                session_ratings = [q['rating'] for q in rag.query_history[-10:]]
                avg_session = sum(session_ratings) / len(session_ratings)
                print(f"   Consultas nesta sessão: {len([q for q in rag.query_history if 'session' not in q])}")
                print(f"   Nota média da sessão: {avg_session:.1f}/5")
            print("\n👋 Obrigado por ajudar o sistema a aprender!")
            break

        # Fazer consulta
        answer, docs_used = rag.query_with_feedback(question)

        # Mostrar resposta
        print("\n🤖 RESPOSTA:")
        print("-" * 60)
        print(answer)
        print("-" * 60)

        # Solicitar feedback
        print("\n⭐ AVALIE A RESPOSTA:")
        print("1 = Muito ruim | 2 = Ruim | 3 = OK | 4 = Boa | 5 = Excelente")

        while True:
            try:
                rating = input("Sua nota (1-5): ").strip()
                rating = int(rating)
                if 1 <= rating <= 5:
                    break
                else:
                    print("Por favor, digite um número entre 1 e 5.")
            except ValueError:
                print("Por favor, digite um número válido.")

        # Registrar feedback
        rag.record_feedback(question, answer, docs_used, rating)

        # Feedback para o usuário
        if rating < 3:
            print("\n🔄 Obrigado! Vou ajustar para melhorar as próximas respostas.")
        elif rating == 3:
            print("\n👍 Obrigado! Continuarei trabalhando para melhorar.")
        else:
            print("\n🎉 Ótimo! Estou aprendendo com seu feedback positivo.")

if __name__ == "__main__":
    main()
