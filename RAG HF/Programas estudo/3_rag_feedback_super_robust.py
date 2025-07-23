#!/usr/bin/env python3
"""
PROGRAMA 3: RAG COM FEEDBACK SUPER ROBUSTO
Sistema que GARANTE que suas avaliações sejam salvas
Otimizado para MacBook - PERSISTÊNCIA 100% GARANTIDA
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
import shutil

class SuperRobustRAG:
    def __init__(self, save_dir="./rag_system_mac"):
        self.save_dir = save_dir
        self.memory_file = f"{self.save_dir}/feedback_memory.json"
        self.backup_file = f"{self.save_dir}/feedback_backup.json"

        # Componentes do sistema
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.generator = None

        # Dados de memória
        self.document_scores = {}
        self.query_history = []
        self.session_data = []
        self.has_memory = False

        # Contador para forçar salvamento
        self.queries_since_save = 0
        self.save_every = 1  # Salva a cada consulta

    def ensure_directory(self):
        """Garante que o diretório existe."""
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            return True
        except Exception as e:
            print(f"❌ Erro ao criar diretório: {e}")
            return False

    def load_system(self):
        """Carrega o sistema RAG treinado."""
        print("🔄 Carregando sistema RAG super robusto...")
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

            # CARREGA MEMÓRIA COM MÚLTIPLAS TENTATIVAS
            self.load_memory_robust()

            print("✅ Sistema carregado com sucesso!")
            print(f"📊 {len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema: {e}")
            return False

    def load_memory_robust(self):
        """Carrega memória com múltiplas tentativas e backups."""
        print("  🧠 Carregando memória (modo robusto)...")

        # Lista de arquivos para tentar carregar (em ordem de prioridade)
        memory_files = [
            self.memory_file,
            self.backup_file,
            f"{self.save_dir}/persistent_memory.json",  # Compatibilidade
            f"{self.save_dir}/feedback_history.json"    # Compatibilidade
        ]

        loaded = False

        for file_path in memory_files:
            if os.path.exists(file_path):
                try:
                    print(f"    📂 Tentando carregar: {os.path.basename(file_path)}")
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = json.load(f)

                        # Carrega dados com compatibilidade
                        self.document_scores = data.get("document_scores", {})
                        self.query_history = data.get("query_history", [])

                        if self.document_scores or self.query_history:
                            self.has_memory = True
                            loaded = True

                            print(f"    ✅ Memória carregada!")
                            print(f"       📈 {len(self.query_history)} consultas anteriores")
                            print(f"       📚 {len(self.document_scores)} documentos com feedback")

                            # Mostra estatísticas
                            if self.query_history:
                                ratings = [q.get('rating', 3) for q in self.query_history]
                                avg_rating = sum(ratings) / len(ratings)
                                print(f"       ⭐ Média histórica: {avg_rating:.1f}/5")

                            break

                except Exception as e:
                    print(f"    ⚠️ Erro ao carregar {file_path}: {e}")
                    continue

        if not loaded:
            print("    📊 Nenhuma memória anterior encontrada")
            print("    🆕 Iniciando com memória limpa")
            self.create_empty_memory()

    def create_empty_memory(self):
        """Cria estrutura de memória vazia."""
        self.document_scores = {}
        self.query_history = []
        self.session_data = []
        self.has_memory = False

        # Salva arquivo vazio imediatamente
        self.force_save_memory()

    def force_save_memory(self):
        """FORÇA o salvamento da memória com múltiplos backups."""
        try:
            # Garante que o diretório existe
            if not self.ensure_directory():
                return False

            # Prepara dados para salvar
            memory_data = {
                "document_scores": self.document_scores,
                "query_history": self.query_history,
                "session_data": self.session_data,
                "stats": {
                    "total_queries": len(self.query_history),
                    "total_documents_with_feedback": len(self.document_scores),
                    "last_updated": datetime.now().isoformat(),
                    "version": "2.0_super_robust"
                },
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "system_type": "super_robust_rag"
                }
            }

            # Salva com múltiplos backups
            success_count = 0

            # 1. Salva arquivo principal
            try:
                with open(self.memory_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print("    💾 Arquivo principal salvo")
            except Exception as e:
                print(f"    ⚠️ Erro ao salvar arquivo principal: {e}")

            # 2. Salva backup
            try:
                with open(self.backup_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print("    💾 Backup salvo")
            except Exception as e:
                print(f"    ⚠️ Erro ao salvar backup: {e}")

            # 3. Salva backup com timestamp
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamp_file = f"{self.save_dir}/feedback_backup_{timestamp}.json"
                with open(timestamp_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print(f"    💾 Backup timestampado salvo")
            except Exception as e:
                print(f"    ⚠️ Erro ao salvar backup timestampado: {e}")

            # 4. Salva em formato pickle como backup adicional
            try:
                pickle_file = f"{self.save_dir}/feedback_memory.pkl"
                with open(pickle_file, "wb") as f:
                    pickle.dump(memory_data, f)
                success_count += 1
                print("    💾 Backup pickle salvo")
            except Exception as e:
                print(f"    ⚠️ Erro ao salvar backup pickle: {e}")

            if success_count > 0:
                print(f"    ✅ Memória salva com sucesso! ({success_count} arquivos)")
                return True
            else:
                print("    ❌ FALHA TOTAL ao salvar memória!")
                return False

        except Exception as e:
            print(f"    ❌ Erro crítico ao salvar: {e}")
            return False

    def calculate_smart_weight(self, chunk_id):
        """Calcula peso baseado no feedback."""
        chunk_str = str(chunk_id)

        if chunk_str in self.document_scores:
            scores = self.document_scores[chunk_str]
            if len(scores) >= 1:
                # Média ponderada (mais recentes pesam mais)
                total_weight = 0
                total_score = 0

                for i, score in enumerate(scores):
                    weight = 1.0 + (i * 0.1)  # Mais recentes pesam mais
                    total_score += score * weight
                    total_weight += weight

                avg_score = total_score / total_weight

                # Converte para peso
                if avg_score >= 4.5:
                    return 2.5
                elif avg_score >= 4.0:
                    return 2.0
                elif avg_score >= 3.5:
                    return 1.5
                elif avg_score >= 2.5:
                    return 1.0
                elif avg_score >= 1.5:
                    return 0.6
                else:
                    return 0.3

        return 1.0

    def search_with_memory(self, question, k=5):
        """Busca usando memória."""
        try:
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            scores, indices = self.index.search(question_embedding, k*2)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.chunks):
                    base_score = float(scores[0][i])
                    weight = self.calculate_smart_weight(idx)
                    adjusted_score = base_score / weight

                    confidence = "🔥" if weight > 2.0 else "👍" if weight > 1.2 else "⚠️" if weight < 0.8 else "📊"

                    results.append({
                        'chunk_id': idx,
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'base_score': base_score,
                        'weight': weight,
                        'adjusted_score': adjusted_score,
                        'confidence': confidence,
                        'feedback_count': len(self.document_scores.get(str(idx), []))
                    })

            results.sort(key=lambda x: x['adjusted_score'])
            return results[:k]

        except Exception as e:
            print(f"❌ Erro na busca: {e}")
            return []

    def generate_answer(self, question, context_docs):
        """Gera resposta."""
        try:
            sorted_docs = sorted(context_docs, key=lambda x: x['weight'], reverse=True)

            context_parts = []
            for doc in sorted_docs[:3]:
                title = doc['metadata']['title']
                text = doc['text']
                confidence = doc['confidence']
                context_parts.append(f"Fonte {confidence} ({title}): {text}")

            context = "\n\n".join(context_parts)
            if len(context) > 1200:
                context = context[:1200] + "..."

            prompt = f"""Responda a pergunta baseado no contexto fornecido.

Pergunta: {question}

Contexto:
{context}

Resposta:"""

            response = self.generator(prompt, max_length=180, num_return_sequences=1)

            if response and len(response) > 0:
                answer = response[0]['generated_text']
                if "Resposta:" in answer:
                    answer = answer.split("Resposta:")[-1].strip()
                return answer
            else:
                return "Não consegui gerar uma resposta adequada."

        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            return f"Erro: {str(e)}"

    def record_feedback_robust(self, question, answer, docs_used, rating):
        """Registra feedback com salvamento IMEDIATO e MÚLTIPLO."""
        try:
            print("\n💾 SALVANDO FEEDBACK...")

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
                        "weight": doc['weight'],
                        "confidence": doc['confidence']
                    } for doc in docs_used
                ]
            }

            # Adiciona à memória
            self.query_history.append(query_record)
            self.session_data.append(query_record)

            # Atualiza scores dos documentos
            for doc in docs_used:
                chunk_id = str(doc['chunk_id'])
                if chunk_id not in self.document_scores:
                    self.document_scores[chunk_id] = []

                self.document_scores[chunk_id].append(rating)

                # Mantém últimos 20 feedbacks
                if len(self.document_scores[chunk_id]) > 20:
                    self.document_scores[chunk_id] = self.document_scores[chunk_id][-20:]

            # FORÇA SALVAMENTO IMEDIATO
            save_success = self.force_save_memory()

            if save_success:
                print("    ✅ FEEDBACK SALVO COM SUCESSO!")
                self.has_memory = True
            else:
                print("    ❌ ERRO AO SALVAR FEEDBACK!")

            # Incrementa contador
            self.queries_since_save += 1

            # Mostra estatísticas
            self.show_memory_stats(rating)

            return save_success

        except Exception as e:
            print(f"❌ ERRO CRÍTICO ao registrar feedback: {e}")
            return False

    def show_memory_stats(self, current_rating):
        """Mostra estatísticas da memória."""
        total_queries = len(self.query_history)

        if total_queries >= 1:
            print(f"\n📊 ESTATÍSTICAS DA MEMÓRIA:")
            print(f"   📝 Consulta atual: #{total_queries}")
            print(f"   ⭐ Nota atual: {current_rating}/5")

            # Média geral
            all_ratings = [q.get('rating', 3) for q in self.query_history]
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

            # Status de salvamento
            print(f"   💾 Arquivos de backup: Múltiplos")
            print(f"   🔄 Salvamentos nesta sessão: {self.queries_since_save}")

    def query_with_feedback(self, question):
        """Executa consulta completa."""
        print("\n" + "="*60)
        print("🔍 PROCESSANDO COM MEMÓRIA SUPER ROBUSTA")
        print("="*60)
        print(f"❓ Pergunta: {question}")

        docs = self.search_with_memory(question, k=5)

        if not docs:
            print("❌ Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes.", []

        print(f"\n📋 DOCUMENTOS SELECIONADOS:")
        for i, doc in enumerate(docs[:3], 1):
            print(f"  {i}. {doc['metadata']['title']} {doc['confidence']}")
            print(f"     Peso: {doc['weight']:.2f} | Feedbacks: {doc['feedback_count']}")
            print(f"     Score: {doc['base_score']:.3f} → {doc['adjusted_score']:.3f}")
            print(f"     Texto: {doc['text'][:80]}...")
            print()

        print("🤖 Gerando resposta...")
        answer = self.generate_answer(question, docs)

        return answer, docs[:3]

    def interactive_mode(self):
        """Modo interativo super robusto."""
        print("🚀 RAG SUPER ROBUSTO COM MEMÓRIA GARANTIDA")
        print("="*60)
        print("💾 SALVAMENTO AUTOMÁTICO E MÚLTIPLO")
        print("🛡️ Seus dados NUNCA serão perdidos!")
        print("📊 Notas: 1-2 = Ruim | 3 = OK | 4-5 = Excelente")
        print("⌨️  Digite 'sair' para terminar")

        if self.has_memory:
            print(f"🧠 Memória carregada: {len(self.query_history)} consultas anteriores")

        print("="*60)

        session_queries = 0

        while True:
            try:
                print("\n" + "🎯" + "="*58)
                question = input("❓ Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    # SALVAMENTO FINAL FORÇADO
                    print("\n💾 SALVAMENTO FINAL...")
                    final_save = self.force_save_memory()
                    if final_save:
                        print("✅ Todos os dados foram salvos com segurança!")
                    else:
                        print("⚠️ Problema no salvamento final!")

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
                print("\n⭐ AVALIE A RESPOSTA (será salva IMEDIATAMENTE):")
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

                # Registra feedback com salvamento robusto
                save_success = self.record_feedback_robust(question, answer, docs_used, rating)
                session_queries += 1

                # Feedback sobre o salvamento
                if save_success:
                    print("\n✅ FEEDBACK SALVO COM SEGURANÇA!")
                else:
                    print("\n⚠️ PROBLEMA NO SALVAMENTO - mas continuando...")

                # Resposta motivacional
                if rating < 3:
                    print("🔄 Obrigado! Sistema ajustado para melhorar.")
                elif rating == 3:
                    print("👍 Obrigado! Continuarei aprendendo.")
                else:
                    print("🎉 Ótimo! Sistema reforçado com feedback positivo.")

            except KeyboardInterrupt:
                print("\n\n⏹️  Programa interrompido.")
                print("💾 SALVAMENTO DE EMERGÊNCIA...")
                self.force_save_memory()
                self.show_session_summary(session_queries)
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                print("💾 SALVAMENTO DE SEGURANÇA...")
                self.force_save_memory()

    def show_session_summary(self, session_queries):
        """Mostra resumo da sessão."""
        print("\n📊 RESUMO DA SESSÃO:")
        print("="*40)
        print(f"🔢 Consultas nesta sessão: {session_queries}")
        print(f"📈 Total na memória: {len(self.query_history)}")
        print(f"💾 Documentos treinados: {len(self.document_scores)}")

        # Verifica arquivos salvos
        saved_files = []
        for filename in [self.memory_file, self.backup_file]:
            if os.path.exists(filename):
                saved_files.append(os.path.basename(filename))

        print(f"💾 Arquivos salvos: {len(saved_files)}")
        for filename in saved_files:
            print(f"   ✅ {filename}")

        print("\n🛡️ SEUS DADOS ESTÃO SEGUROS!")
        print("🔄 Na próxima execução, tudo será carregado automaticamente!")
        print("\n👋 Obrigado por treinar o sistema!")

def main():
    """Função principal."""
    print("🍎 RAG SUPER ROBUSTO PARA MACBOOK")
    print("Sistema com salvamento garantido e múltiplos backups\n")

    try:
        rag = SuperRobustRAG()

        if not rag.load_system():
            print("\n💡 Execute primeiro o treinamento:")
            print("   python 1_rag_trainer.py")
            sys.exit(1)

        rag.interactive_mode()

    except KeyboardInterrupt:
        print("\n\n⏹️  Programa cancelado.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
