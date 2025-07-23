#!/usr/bin/env python3
"""
PROGRAMA 3: RAG COM FEEDBACK PERSISTENTE
Sistema RAG que aprende e LEMBRA das avaliações do usuário
Otimizado para MacBook - MEMÓRIA PERMANENTE
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

class SmartRAGWithMemory:
    def __init__(self, save_dir="./rag_system_mac"):
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.generator = None

        # Sistema de memória persistente
        self.feedback_data = {}
        self.document_scores = {}
        self.query_history = []
        self.learning_stats = {
            "total_queries": 0,
            "avg_rating": 0.0,
            "best_documents": {},
            "worst_documents": {},
            "improvement_trend": []
        }

    def load_system(self):
        """Carrega o sistema RAG treinado."""
        print("🔄 Carregando sistema RAG com memória...")
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

            # CARREGA MEMÓRIA PERSISTENTE
            self.load_persistent_memory()

            print("✅ Sistema carregado com sucesso!")
            print(f"📊 {len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def load_persistent_memory(self):
        """Carrega TODA a memória de feedback anterior."""
        try:
            feedback_file = f"{self.save_dir}/persistent_memory.json"

            if os.path.exists(feedback_file):
                print("  🧠 Carregando memória persistente...")
                with open(feedback_file, "r", encoding='utf-8') as f:
                    memory_data = json.load(f)

                    self.feedback_data = memory_data.get("feedback_data", {})
                    self.document_scores = memory_data.get("document_scores", {})
                    self.query_history = memory_data.get("query_history", [])
                    self.learning_stats = memory_data.get("learning_stats", {
                        "total_queries": 0,
                        "avg_rating": 0.0,
                        "best_documents": {},
                        "worst_documents": {},
                        "improvement_trend": []
                    })

                # Mostra estatísticas da memória carregada
                total_queries = len(self.query_history)
                docs_with_feedback = len(self.document_scores)

                print(f"  📈 Memória carregada:")
                print(f"     🔢 {total_queries} consultas anteriores")
                print(f"     📚 {docs_with_feedback} documentos com feedback")

                if total_queries > 0:
                    recent_ratings = [q['rating'] for q in self.query_history[-10:]]
                    if recent_ratings:
                        avg_recent = sum(recent_ratings) / len(recent_ratings)
                        print(f"     ⭐ Média últimas consultas: {avg_recent:.1f}/5")

                # Mostra documentos mais bem avaliados
                if self.document_scores:
                    best_docs = self.get_best_documents(3)
                    if best_docs:
                        print(f"     🏆 Melhores documentos:")
                        for doc_id, avg_score in best_docs:
                            title = self.metadata[int(doc_id)]['title']
                            print(f"        - {title}: {avg_score:.1f}/5")

                print("  ✅ Memória restaurada com sucesso!")

            else:
                print("  📊 Nenhuma memória anterior - começando do zero")
                self.create_empty_memory_file()

        except Exception as e:
            print(f"  ⚠️ Erro ao carregar memória: {e}")
            print("  🔄 Criando nova memória...")
            self.create_empty_memory_file()

    def create_empty_memory_file(self):
        """Cria arquivo de memória vazio."""
        try:
            empty_memory = {
                "feedback_data": {},
                "document_scores": {},
                "query_history": [],
                "learning_stats": {
                    "total_queries": 0,
                    "avg_rating": 0.0,
                    "best_documents": {},
                    "worst_documents": {},
                    "improvement_trend": []
                },
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            }

            with open(f"{self.save_dir}/persistent_memory.json", "w", encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"  ⚠️ Erro ao criar arquivo de memória: {e}")

    def save_persistent_memory(self):
        """Salva TODA a memória de forma persistente."""
        try:
            feedback_file = f"{self.save_dir}/persistent_memory.json"

            # Atualiza estatísticas
            self.update_learning_stats()

            # Prepara dados completos para salvar
            memory_data = {
                "feedback_data": self.feedback_data,
                "document_scores": self.document_scores,
                "query_history": self.query_history,
                "learning_stats": self.learning_stats,
                "last_updated": datetime.now().isoformat(),
                "total_queries": len(self.query_history),
                "total_documents_with_feedback": len(self.document_scores),
                "version": "1.0"
            }

            # Salva com backup
            backup_file = f"{self.save_dir}/persistent_memory_backup.json"
            if os.path.exists(feedback_file):
                # Cria backup da versão anterior
                os.rename(feedback_file, backup_file)

            # Salva nova versão
            with open(feedback_file, "w", encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)

            # Remove backup se salvou com sucesso
            if os.path.exists(backup_file):
                os.remove(backup_file)

            return True

        except Exception as e:
            print(f"⚠️ Erro ao salvar memória: {e}")
            # Restaura backup se houver erro
            backup_file = f"{self.save_dir}/persistent_memory_backup.json"
            if os.path.exists(backup_file):
                os.rename(backup_file, feedback_file)
            return False

    def update_learning_stats(self):
        """Atualiza estatísticas de aprendizado."""
        if not self.query_history:
            return

        # Calcula média geral
        all_ratings = [q['rating'] for q in self.query_history]
        self.learning_stats["total_queries"] = len(all_ratings)
        self.learning_stats["avg_rating"] = sum(all_ratings) / len(all_ratings)

        # Calcula tendência de melhoria (últimas 10 vs primeiras 10)
        if len(all_ratings) >= 20:
            first_10 = sum(all_ratings[:10]) / 10
            last_10 = sum(all_ratings[-10:]) / 10
            improvement = last_10 - first_10
            self.learning_stats["improvement_trend"].append({
                "timestamp": datetime.now().isoformat(),
                "improvement": improvement,
                "first_10_avg": first_10,
                "last_10_avg": last_10
            })

        # Identifica melhores e piores documentos
        best_docs = self.get_best_documents(5)
        worst_docs = self.get_worst_documents(5)

        self.learning_stats["best_documents"] = {str(doc_id): score for doc_id, score in best_docs}
        self.learning_stats["worst_documents"] = {str(doc_id): score for doc_id, score in worst_docs}

    def get_best_documents(self, n=5):
        """Retorna os N melhores documentos por avaliação."""
        doc_averages = []
        for doc_id, scores in self.document_scores.items():
            if len(scores) >= 2:  # Pelo menos 2 avaliações
                avg_score = sum(scores) / len(scores)
                doc_averages.append((doc_id, avg_score))

        doc_averages.sort(key=lambda x: x[1], reverse=True)
        return doc_averages[:n]

    def get_worst_documents(self, n=5):
        """Retorna os N piores documentos por avaliação."""
        doc_averages = []
        for doc_id, scores in self.document_scores.items():
            if len(scores) >= 2:  # Pelo menos 2 avaliações
                avg_score = sum(scores) / len(scores)
                doc_averages.append((doc_id, avg_score))

        doc_averages.sort(key=lambda x: x[1])
        return doc_averages[:n]

    def calculate_smart_weight(self, chunk_id):
        """Calcula peso inteligente baseado no histórico completo."""
        chunk_str = str(chunk_id)

        if chunk_str in self.document_scores:
            scores = self.document_scores[chunk_str]
            if len(scores) >= 1:
                # Calcula média ponderada (avaliações mais recentes têm peso maior)
                weighted_scores = []
                for i, score in enumerate(scores):
                    weight = 1.0 + (i * 0.1)  # Avaliações mais recentes pesam mais
                    weighted_scores.append(score * weight)

                avg_score = sum(weighted_scores) / sum(1.0 + (i * 0.1) for i in range(len(scores)))

                # Converte para peso (1-5 -> 0.2-2.5)
                if avg_score >= 4.0:
                    return 2.5  # Documentos excelentes
                elif avg_score >= 3.5:
                    return 1.8  # Documentos bons
                elif avg_score >= 2.5:
                    return 1.0  # Documentos neutros
                elif avg_score >= 1.5:
                    return 0.6  # Documentos ruins
                else:
                    return 0.2  # Documentos muito ruins

        return 1.0  # Peso neutro para documentos sem feedback

    def search_with_memory(self, question, k=5):
        """Busca documentos usando TODA a memória de feedback."""
        try:
            # Cria embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Busca mais documentos para ter opções
            scores, indices = self.index.search(question_embedding, k*3)

            # Aplica pesos baseados na memória completa
            weighted_results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    base_score = float(scores[0][i])
                    memory_weight = self.calculate_smart_weight(idx)
                    adjusted_score = base_score / memory_weight  # Menor score = melhor

                    # Calcula confiança baseada no histórico
                    chunk_str = str(idx)
                    confidence = "🔥" if memory_weight > 2.0 else "👍" if memory_weight > 1.2 else "⚠️" if memory_weight < 0.8 else "📊"

                    weighted_results.append({
                        'chunk_id': idx,
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'base_score': base_score,
                        'memory_weight': memory_weight,
                        'adjusted_score': adjusted_score,
                        'confidence': confidence,
                        'feedback_count': len(self.document_scores.get(chunk_str, [])),
                        'rank': i + 1
                    })

            # Ordena por score ajustado e pega os melhores
            weighted_results.sort(key=lambda x: x['adjusted_score'])
            return weighted_results[:k]

        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []

    def generate_memory_based_answer(self, question, context_docs):
        """Gera resposta priorizando documentos com melhor histórico."""
        try:
            # Ordena documentos por peso de memória (melhores primeiro)
            sorted_docs = sorted(context_docs, key=lambda x: x['memory_weight'], reverse=True)

            # Cria contexto priorizando documentos bem avaliados
            context_parts = []
            for doc in sorted_docs[:3]:  # Usa os 3 melhores
                title = doc['metadata']['title']
                text = doc['text']
                confidence = doc['confidence']
                context_parts.append(f"Fonte {confidence} ({title}): {text}")

            context = "\n\n".join(context_parts)

            # Limita tamanho do contexto
            if len(context) > 1200:
                context = context[:1200] + "..."

            # Cria prompt otimizado baseado na memória
            prompt = f"""Com base no contexto fornecido, responda a pergunta de forma clara e precisa.

Pergunta: {question}

Contexto (fontes verificadas):
{context}

Resposta detalhada:"""

            # Gera resposta
            response = self.generator(prompt, max_length=180, num_return_sequences=1)

            if response and len(response) > 0:
                answer = response[0]['generated_text']
                # Limpa a resposta
                if "Resposta detalhada:" in answer:
                    answer = answer.split("Resposta detalhada:")[-1].strip()
                elif "Resposta:" in answer:
                    answer = answer.split("Resposta:")[-1].strip()
                return answer
            else:
                return "Não consegui gerar uma resposta adequada com base na memória atual."

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return f"Erro: {str(e)}"

    def record_persistent_feedback(self, question, answer, docs_used, rating):
        """Registra feedback de forma PERMANENTE."""
        try:
            # Cria registro da consulta
            query_record = {
                "id": f"query_{len(self.query_history)}",
                "question": question,
                "answer": answer,
                "rating": rating,
                "timestamp": datetime.now().isoformat(),
                "docs_used": [
                    {
                        "chunk_id": doc['chunk_id'],
                        "title": doc['metadata']['title'],
                        "memory_weight": doc['memory_weight'],
                        "confidence": doc['confidence']
                    } for doc in docs_used
                ]
            }
            self.query_history.append(query_record)

            # Atualiza scores dos documentos PERMANENTEMENTE
            for doc in docs_used:
                chunk_id = str(doc['chunk_id'])
                if chunk_id not in self.document_scores:
                    self.document_scores[chunk_id] = []

                self.document_scores[chunk_id].append(rating)

                # Mantém histórico mais longo (últimos 25 feedbacks)
                if len(self.document_scores[chunk_id]) > 25:
                    self.document_scores[chunk_id] = self.document_scores[chunk_id][-25:]

            # SALVA IMEDIATAMENTE na memória persistente
            save_success = self.save_persistent_memory()

            if save_success:
                print("💾 Feedback salvo permanentemente!")
            else:
                print("⚠️ Aviso: Erro ao salvar feedback")

            # Mostra estatísticas atualizadas
            self.show_memory_stats(rating)

        except Exception as e:
            print(f"⚠️ Erro ao registrar feedback: {e}")

    def show_memory_stats(self, current_rating):
        """Mostra estatísticas baseadas na memória completa."""
        total_queries = len(self.query_history)

        if total_queries >= 1:
            print(f"\n🧠 ESTATÍSTICAS DA MEMÓRIA:")
            print(f"   📝 Consulta atual: #{total_queries}")
            print(f"   ⭐ Nota atual: {current_rating}/5")

            # Média geral
            all_ratings = [q['rating'] for q in self.query_history]
            avg_all = sum(all_ratings) / len(all_ratings)
            print(f"   📊 Média geral: {avg_all:.1f}/5")

            # Últimas consultas
            if total_queries > 1:
                recent_count = min(5, total_queries)
                recent_ratings = all_ratings[-recent_count:]
                avg_recent = sum(recent_ratings) / len(recent_ratings)
                print(f"   📈 Média últimas {recent_count}: {avg_recent:.1f}/5")

            # Documentos com feedback
            docs_with_feedback = len(self.document_scores)
            print(f"   📚 Documentos treinados: {docs_with_feedback}")

            # Tendência de melhoria
            if total_queries >= 10:
                first_5 = sum(all_ratings[:5]) / 5
                last_5 = sum(all_ratings[-5:]) / 5
                improvement = last_5 - first_5
                trend_emoji = "📈" if improvement > 0.2 else "📊" if improvement > -0.2 else "📉"
                print(f"   {trend_emoji} Tendência: {improvement:+.1f} pontos")

            # Feedback motivacional baseado na memória
            if current_rating >= 4:
                print("   🎉 Excelente! A memória está sendo atualizada.")
            elif current_rating >= 3:
                print("   👍 Boa resposta. Sistema aprendendo continuamente.")
            else:
                print("   🔄 Resposta ruim. Ajustando memória para melhorar.")

    def query_with_memory(self, question):
        """Executa consulta usando TODA a memória de feedback."""
        print("\n" + "="*60)
        print("🔍 PROCESSANDO COM MEMÓRIA COMPLETA")
        print("="*60)
        print(f"❓ Pergunta: {question}")

        # Busca documentos usando memória
        print("\n🧠 Buscando com base na memória de feedback...")
        docs = self.search_with_memory(question, k=5)

        if not docs:
            print("❌ Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes.", []

        # Mostra documentos selecionados com informações de memória
        print(f"\n📋 DOCUMENTOS SELECIONADOS (baseado na memória):")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. {doc['metadata']['title']} {doc['confidence']}")
            print(f"     Peso memória: {doc['memory_weight']:.2f} | Feedbacks: {doc['feedback_count']}")
            print(f"     Score: {doc['base_score']:.3f} → {doc['adjusted_score']:.3f}")
            print(f"     Texto: {doc['text'][:80]}...")
            print()

        # Gera resposta baseada na memória
        print("🤖 Gerando resposta baseada na memória...")
        answer = self.generate_memory_based_answer(question, docs)

        return answer, docs[:3]

    def interactive_memory_mode(self):
        """Modo interativo com memória persistente completa."""
        print("🚀 RAG COM MEMÓRIA PERSISTENTE")
        print("="*60)
        print("🧠 Este sistema LEMBRA de todas suas avaliações!")
        print("💾 Feedback é salvo permanentemente")
        print("📊 Notas: 1-2 = Ruim | 3 = OK | 4-5 = Excelente")
        print("⌨️  Digite 'sair' para terminar")

        # Mostra estatísticas da memória carregada
        if self.query_history:
            total = len(self.query_history)
            avg = sum(q['rating'] for q in self.query_history) / total
            print(f"🔄 Memória carregada: {total} consultas anteriores (média: {avg:.1f}/5)")

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

                # Processa pergunta com memória
                answer, docs_used = self.query_with_memory(question)

                # Mostra resposta
                print("\n🤖 RESPOSTA (baseada na memória):")
                print("-" * 60)
                print(answer)
                print("-" * 60)

                # Solicita avaliação
                print("\n⭐ AVALIE A RESPOSTA (será salva permanentemente):")
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

                # Registra feedback PERMANENTEMENTE
                self.record_persistent_feedback(question, answer, docs_used, rating)
                session_queries += 1

                # Resposta motivacional
                if rating < 3:
                    print("\n🔄 Obrigado! Memória atualizada para melhorar.")
                elif rating == 3:
                    print("\n👍 Obrigado! Continuarei aprendendo com a memória.")
                else:
                    print("\n🎉 Ótimo! Memória reforçada com feedback positivo.")

            except KeyboardInterrupt:
                print("\n\n⏹️  Programa interrompido.")
                self.show_session_summary(session_queries)
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")

    def show_session_summary(self, session_queries):
        """Mostra resumo da sessão com estatísticas de memória."""
        print("\n📊 RESUMO DA SESSÃO:")
        print("="*40)
        print(f"🔢 Consultas nesta sessão: {session_queries}")
        print(f"📈 Total na memória: {len(self.query_history)}")
        print(f"💾 Documentos treinados: {len(self.document_scores)}")

        if self.query_history:
            # Estatísticas da sessão
            if session_queries > 0:
                recent_ratings = [q['rating'] for q in self.query_history[-session_queries:]]
                avg_session = sum(recent_ratings) / len(recent_ratings)
                print(f"⭐ Nota média da sessão: {avg_session:.1f}/5")

            # Estatísticas gerais da memória
            all_ratings = [q['rating'] for q in self.query_history]
            avg_all = sum(all_ratings) / len(all_ratings)
            print(f"🧠 Média geral da memória: {avg_all:.1f}/5")

            # Melhores documentos
            best_docs = self.get_best_documents(3)
            if best_docs:
                print("\n🏆 Documentos mais bem avaliados:")
                for doc_id, score in best_docs:
                    title = self.metadata[int(doc_id)]['title']
                    print(f"   - {title}: {score:.1f}/5")

        print("\n💾 Toda sua avaliação foi salva permanentemente!")
        print("🔄 Na próxima execução, o sistema lembrará de tudo!")
        print("\n👋 Obrigado por treinar o sistema!")

def main():
    """Função principal."""
    print("🍎 RAG COM MEMÓRIA PERSISTENTE PARA MACBOOK")
    print("Sistema que aprende e LEMBRA para sempre\n")

    try:
        # Inicializa sistema
        rag = SmartRAGWithMemory()

        # Carrega sistema
        if not rag.load_system():
            print("\n💡 Execute primeiro o treinamento:")
            print("   python 1_rag_trainer.py")
            sys.exit(1)

        # Inicia modo interativo com memória
        rag.interactive_memory_mode()

    except KeyboardInterrupt:
        print("\n\n⏹️  Programa cancelado.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
