#!/usr/bin/env python3
"""
PROGRAMA 3: RAG COM FEEDBACK SUPER ROBUSTO
Compatível com arquivos do 1_rag_trainer.py
"""

import pickle
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime
import sys
import warnings
import torch

# Suprimir warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

class SuperRobustRAG:
    def __init__(self, save_dir="./rag_system"):
        self.save_dir = save_dir
        self.memory_file = f"{self.save_dir}/feedback_memory.json"
        self.backup_file = f"{self.save_dir}/feedback_backup.json"

        # Componentes do sistema
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None

        # Dados de memória
        self.document_scores = {}
        self.query_history = []
        self.session_data = []
        self.has_memory = False

        # Contador para forçar salvamento
        self.queries_since_save = 0
        self.save_every = 1  # Salva a cada consulta

    def ensure_directory(self):
        # Garante que o diretório existe.
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            return True
        except Exception as e:
            print(f"Erro ao criar diretório: {e}")
            return False

    def load_system(self):
        # Carrega o sistema RAG treinado.
        print("Carregando sistema RAG...")
        print(f"Diretório: {os.path.abspath(self.save_dir)}")

        if not os.path.exists(self.save_dir):
            print(f"Sistema não encontrado em {self.save_dir}")
            print("Execute primeiro: python 1_rag_trainer_corrigido.py")
            return False

        try:
            # Carrega componentes básicos - COMPATÍVEL COM TRAINER CORRIGIDO
            print("  Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            print("  Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            print("  Carregando dados...")
            with open(f"{self.save_dir}/rag_data.pkl", "rb") as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']

            # CARREGA MEMÓRIA COM MÚLTIPLAS TENTATIVAS
            self.load_memory_robust()

            print("Sistema carregado com sucesso!")
            print(f"{len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"Erro ao carregar sistema: {e}")
            return False

    def load_memory_robust(self):
        """Carrega memória com múltiplas tentativas e backups."""
        print("  Carregando memória (modo robusto)...")

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
                    print(f"    Tentando carregar: {os.path.basename(file_path)}")
                    with open(file_path, "r", encoding='utf-8') as f:
                        data = json.load(f)

                    # Carrega dados com compatibilidade
                    self.document_scores = data.get("document_scores", {})
                    self.query_history = data.get("query_history", [])

                    if self.document_scores or self.query_history:
                        self.has_memory = True
                        loaded = True

                    print(f"    Memória carregada!")
                    print(f"    {len(self.query_history)} consultas anteriores")
                    print(f"    {len(self.document_scores)} documentos com feedback")

                    # Mostra estatísticas
                    if self.query_history:
                        ratings = [q.get('rating', 3) for q in self.query_history]
                        avg_rating = sum(ratings) / len(ratings)
                        print(f"    Média histórica: {avg_rating:.1f}/5")

                    break

                except Exception as e:
                    print(f"    Erro ao carregar {file_path}: {e}")
                    continue

        if not loaded:
            print("    Nenhuma memória anterior encontrada")
            print("    Iniciando com memória limpa")
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

            # Prepara dados para salvar - CORRIGIDO: conversão de tipos numpy
            memory_data = {
                "document_scores": {},
                "query_history": [],
                "session_data": [],
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

            # Converter document_scores (numpy types para tipos nativos)
            for key, value in self.document_scores.items():
                if isinstance(value, list):
                    memory_data["document_scores"][key] = [int(v) if isinstance(v, (np.int64, np.int32)) else float(v) if isinstance(v, (np.float64, np.float32)) else v for v in value]
                else:
                    memory_data["document_scores"][key] = value

            # Converter query_history
            for query in self.query_history:
                query_clean = {}
                for k, v in query.items():
                    if isinstance(v, (np.int64, np.int32)):
                        query_clean[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        query_clean[k] = float(v)
                    elif isinstance(v, list):
                        query_clean[k] = [int(item) if isinstance(item, (np.int64, np.int32)) else float(item) if isinstance(item, (np.float64, np.float32)) else item for item in v]
                    else:
                        query_clean[k] = v
                memory_data["query_history"].append(query_clean)

            # Converter session_data
            for session in self.session_data:
                session_clean = {}
                for k, v in session.items():
                    if isinstance(v, (np.int64, np.int32)):
                        session_clean[k] = int(v)
                    elif isinstance(v, (np.float64, np.float32)):
                        session_clean[k] = float(v)
                    elif isinstance(v, list):
                        session_clean[k] = [int(item) if isinstance(item, (np.int64, np.int32)) else float(item) if isinstance(item, (np.float64, np.float32)) else item for item in v]
                    else:
                        session_clean[k] = v
                memory_data["session_data"].append(session_clean)

            # Salva com múltiplos backups
            success_count = 0

            # 1. Salva arquivo principal
            try:
                with open(self.memory_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print("    Arquivo principal salvo")
            except Exception as e:
                print(f"    Erro ao salvar arquivo principal: {e}")

            # 2. Salva backup
            try:
                with open(self.backup_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print("    Backup salvo")
            except Exception as e:
                print(f"    Erro ao salvar backup: {e}")

            # 3. Salva backup com timestamp
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamp_file = f"{self.save_dir}/feedback_backup_{timestamp}.json"
                with open(timestamp_file, "w", encoding='utf-8') as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)
                success_count += 1
                print(f"    Backup timestampado salvo")
            except Exception as e:
                print(f"    Erro ao salvar backup timestampado: {e}")

            if success_count > 0:
                print(f"    Memória salva com sucesso! ({success_count} arquivos)")
                return True
            else:
                print("    FALHA TOTAL ao salvar memória!")
                return False

        except Exception as e:
            print(f"    Erro crítico ao salvar: {e}")
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

                    confidence = "Excelente" if weight > 2.0 else "Bom" if weight > 1.2 else "Baixo" if weight < 0.8 else "Médio"

                    results.append({
                        'chunk_id': int(idx),  # Converter para int nativo
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
            print(f"Erro na busca: {e}")
            return []

    def generate_consolidated_answer(self, question, context_docs):
        """Gera resposta consolidada e inteligente baseada no contexto."""
        try:
            if not context_docs:
                return "Desculpe, não encontrei informações relevantes para sua pergunta."

            # Ordena documentos por relevância (peso + score)
            sorted_docs = sorted(context_docs, key=lambda x: x['weight'] * (1 / (x['adjusted_score'] + 0.001)), reverse=True)

            # Coleta informações dos melhores documentos
            all_info = []
            sources = []

            for doc in sorted_docs[:3]:  # Usa os 3 melhores documentos
                text = doc['text']
                title = doc['metadata']['title']

                # Adiciona informação e fonte
                all_info.append(text)
                if title not in sources:
                    sources.append(title)

            # Combina todas as informações
            combined_text = " ".join(all_info)

            # Gera resposta consolidada baseada na pergunta
            answer = self.create_intelligent_response(question, combined_text, sources)

            return answer

        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return f"Erro ao processar sua pergunta: {str(e)}"

    def create_intelligent_response(self, question, combined_text, sources):
        """Cria uma resposta inteligente e consolidada."""
        try:
            # Limita o texto para evitar respostas muito longas
            if len(combined_text) > 1500:
                combined_text = combined_text[:1500] + "..."

            # Identifica palavras-chave da pergunta
            question_lower = question.lower()
            key_words = [word for word in question_lower.split() if len(word) > 3]

            # Encontra sentenças mais relevantes
            sentences = combined_text.split('.')
            relevant_sentences = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Ignora sentenças muito curtas
                    sentence_lower = sentence.lower()
                    # Verifica se a sentença contém palavras-chave da pergunta
                    relevance_score = sum(1 for word in key_words if word in sentence_lower)
                    if relevance_score > 0:
                        relevant_sentences.append((sentence, relevance_score))

            # Ordena por relevância e pega as melhores
            relevant_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentences = [sent[0] for sent in relevant_sentences[:4]]

            if best_sentences:
                # Constrói resposta consolidada
                answer = f"Com base nas informações encontradas sobre '{question}':\n\n"

                # Adiciona as informações mais relevantes
                for i, sentence in enumerate(best_sentences, 1):
                    if sentence.endswith('.'):
                        answer += f"{sentence} "
                    else:
                        answer += f"{sentence}. "

                # Adiciona fontes
                if sources:
                    answer += f"\n\nFontes consultadas: {', '.join(sources[:3])}"

                return answer.strip()
            else:
                # Fallback: resposta simples baseada no texto
                answer = f"Sobre '{question}': "
                answer += combined_text[:600] + "..."

                if sources:
                    answer += f"\n\nFonte: {sources[0]}"

                return answer

        except Exception as e:
            print(f"Erro na criação da resposta: {e}")
            return f"Com base nas informações encontradas: {combined_text[:400]}..."

    def record_feedback_robust(self, question, answer, docs_used, rating):
        """Registra feedback com salvamento IMEDIATO e MÚLTIPLO."""
        try:
            print("\nSALVANDO FEEDBACK...")

            # Cria registro da consulta
            query_record = {
                "id": f"query_{len(self.query_history)}",
                "question": question,
                "answer": answer,
                "rating": int(rating),  # Garantir que é int nativo
                "timestamp": datetime.now().isoformat(),
                "docs_used": [
                    {
                        "chunk_id": int(doc['chunk_id']),  # Converter para int nativo
                        "title": doc['metadata']['title'],
                        "weight": float(doc['weight']),  # Converter para float nativo
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

                self.document_scores[chunk_id].append(int(rating))  # Garantir que é int nativo

                # Mantém últimos 20 feedbacks
                if len(self.document_scores[chunk_id]) > 20:
                    self.document_scores[chunk_id] = self.document_scores[chunk_id][-20:]

            # FORÇA SALVAMENTO IMEDIATO
            save_success = self.force_save_memory()

            if save_success:
                print("    FEEDBACK SALVO COM SUCESSO!")
                self.has_memory = True
            else:
                print("    ERRO AO SALVAR FEEDBACK!")

            # Incrementa contador
            self.queries_since_save += 1

            # Mostra estatísticas
            self.show_memory_stats(rating)

            return save_success

        except Exception as e:
            print(f"ERRO CRÍTICO ao registrar feedback: {e}")
            return False

    def show_memory_stats(self, current_rating):
        """Mostra estatísticas da memória."""
        total_queries = len(self.query_history)

        if total_queries >= 1:
            print(f"\nESTATÍSTICAS DA MEMÓRIA:")
            print(f"   Consulta atual: #{total_queries}")
            print(f"   Nota atual: {current_rating}/5")

            # Média geral
            all_ratings = [q.get('rating', 3) for q in self.query_history]
            avg_all = sum(all_ratings) / len(all_ratings)
            print(f"   Média geral: {avg_all:.1f}/5")

            # Últimas consultas
            if total_queries > 1:
                recent_count = min(5, total_queries)
                recent_ratings = all_ratings[-recent_count:]
                avg_recent = sum(recent_ratings) / len(recent_ratings)
                print(f"   Média últimas {recent_count}: {avg_recent:.1f}/5")

            # Documentos com feedback
            docs_with_feedback = len(self.document_scores)
            print(f"   Documentos treinados: {docs_with_feedback}")

    def query_with_feedback(self, question):
        """Executa consulta completa com resposta consolidada."""
        print("\n" + "="*60)
        print("PROCESSANDO CONSULTA COM INTELIGÊNCIA CONSOLIDADA")
        print("="*60)
        print(f"Pergunta: {question}")

        docs = self.search_with_memory(question, k=5)

        if not docs:
            print("Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes.", []

        print(f"\nDocumentos encontrados: {len(docs)}")
        print("Gerando resposta consolidada...")

        # Gera resposta consolidada (não mostra mais os documentos individuais)
        answer = self.generate_consolidated_answer(question, docs)

        return answer, docs[:3]

    def interactive_mode(self):
        """Modo interativo super robusto."""
        print("RAG SUPER ROBUSTO COM RESPOSTA CONSOLIDADA")
        print("="*60)
        print("SALVAMENTO AUTOMÁTICO E MÚLTIPLO")
        print("Respostas inteligentes e consolidadas!")
        print("Notas: 1-2 = Ruim | 3 = OK | 4-5 = Excelente")
        print("Digite 'sair' para terminar")

        if self.has_memory:
            print(f"Memória carregada: {len(self.query_history)} consultas anteriores")

        print("="*60)

        session_queries = 0

        while True:
            try:
                print("\n" + "="*58)
                question = input("Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    # SALVAMENTO FINAL FORÇADO
                    print("\nSALVAMENTO FINAL...")
                    final_save = self.force_save_memory()
                    if final_save:
                        print("Todos os dados foram salvos com segurança!")
                    else:
                        print("Problema no salvamento final!")

                    self.show_session_summary(session_queries)
                    break

                if len(question) < 3:
                    print("Pergunta muito curta. Tente novamente.")
                    continue

                # Processa pergunta
                answer, docs_used = self.query_with_feedback(question)

                # Mostra resposta consolidada
                print("\nRESPOSTA CONSOLIDADA:")
                print("-" * 60)
                print(answer)
                print("-" * 60)

                # Solicita avaliação
                print("\nAVALIE A RESPOSTA (será salva IMEDIATAMENTE):")
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
                    print("\nFEEDBACK SALVO COM SEGURANÇA!")
                else:
                    print("\nPROBLEMA NO SALVAMENTO - mas continuando...")

                # Resposta motivacional
                if rating < 3:
                    print("Obrigado! Sistema ajustado para melhorar.")
                elif rating == 3:
                    print("Obrigado! Continuarei aprendendo.")
                else:
                    print("Ótimo! Sistema reforçado com feedback positivo.")

            except KeyboardInterrupt:
                print("\n\nPrograma interrompido.")
                print("SALVAMENTO DE EMERGÊNCIA...")
                self.force_save_memory()
                self.show_session_summary(session_queries)
                break
            except Exception as e:
                print(f"\nErro: {e}")
                print("SALVAMENTO DE SEGURANÇA...")
                self.force_save_memory()

    def show_session_summary(self, session_queries):
        """Mostra resumo da sessão."""
        print("\nRESUMO DA SESSÃO:")
        print("="*40)
        print(f"Consultas nesta sessão: {session_queries}")
        print(f"Total na memória: {len(self.query_history)}")
        print(f"Documentos treinados: {len(self.document_scores)}")

        # Verifica arquivos salvos
        saved_files = []
        for filename in [self.memory_file, self.backup_file]:
            if os.path.exists(filename):
                saved_files.append(os.path.basename(filename))

        print(f"Arquivos salvos: {len(saved_files)}")
        for filename in saved_files:
            print(f"   {filename}")

        print("\nSEUS DADOS ESTÃO SEGUROS!")
        print("Na próxima execução, tudo será carregado automaticamente!")
        print("\nObrigado por treinar o sistema!")

def main():
    """Função principal."""
    print("RAG SUPER ROBUSTO - VERSÃO CONSOLIDADA")
    print("Sistema com respostas inteligentes e consolidadas\n")

    try:
        rag = SuperRobustRAG()

        if not rag.load_system():
            print("\nExecute primeiro o treinamento:")
            print("   python 1_rag_trainer_corrigido.py")
            sys.exit(1)

        rag.interactive_mode()

    except KeyboardInterrupt:
        print("\n\nPrograma cancelado.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
