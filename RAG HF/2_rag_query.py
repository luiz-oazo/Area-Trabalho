#!/usr/bin/env python3
"""
PROGRAMA 2: CONSULTA RAG INTELIGENTE ULTRA ROBUSTA
Usa o aprendizado do programa 3 para dar respostas melhores
"""

import pickle
import os
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import sys
import warnings
import torch

# Suprimir warnings
warnings.filterwarnings("ignore")
torch.set_warn_always(False)

class UltraRobustRAGQuery:
    def __init__(self, save_dir="./rag_system"):
        self.save_dir = save_dir
        self.embedding_model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.use_hf_generator = False
        self.generator = None

        # Sistema de memória (carregado do programa 3)
        self.document_scores = {}
        self.has_memory = False
        self.total_feedback_queries = 0

    def load_system(self):
        # Carrega o sistema RAG treinado.
        #print("Carregando sistema RAG inteligente...")
        #print(f"Diretório: {os.path.abspath(self.save_dir)}")

        if not os.path.exists(self.save_dir):
            #print(f"Sistema não encontrado em {self.save_dir}")
            #print("Execute primeiro: python 1_rag_trainer_corrigido.py")
            return False

        try:
            # Carrega componentes básicos
            #print("  Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            #print("  Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            #print("  Carregando dados...")
            with open(f"{self.save_dir}/rag_data.pkl", "rb") as f:
                data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']

            # MODO SEGURO: Não carrega HuggingFace por padrão
            #print("  Modo gerador: INTELIGENTE SEM HF (evita segmentation fault)")
            #print("    Usando algoritmo próprio de consolidação")

            # TENTA CARREGAR MEMÓRIA DO PROGRAMA 3
            self.load_feedback_memory()

            #print("Sistema carregado com sucesso!")
            #print(f"{len(self.chunks)} chunks disponíveis")
            return True

        except Exception as e:
            print(f"Erro ao carregar sistema: {e}")
            return False

    def load_feedback_memory(self):
        # Carrega memória de feedback do programa 3 (compatível com todas as versões).
        try:
            # Lista de arquivos de memória para tentar (em ordem de prioridade)
            memory_files = [
                f"{self.save_dir}/feedback_memory.json",      # Versão super robust
                f"{self.save_dir}/feedback_backup.json",     # Backup super robust
                f"{self.save_dir}/persistent_memory.json",   # Versão original
                f"{self.save_dir}/feedback_history.json"     # Compatibilidade extra
            ]

            loaded = False

            for memory_file in memory_files:
                if os.path.exists(memory_file):
                    try:
                        #print(f"  Carregando memória de: {os.path.basename(memory_file)}")
                        with open(memory_file, "r", encoding='utf-8') as f:
                            memory_data = json.load(f)

                        self.document_scores = memory_data.get("document_scores", {})
                        query_history = memory_data.get("query_history", [])
                        self.total_feedback_queries = len(query_history)

                        if self.document_scores or query_history:
                            self.has_memory = True
                            docs_with_feedback = len(self.document_scores)

                            #print(f"  Memória encontrada!")
                            #print(f"    {self.total_feedback_queries} consultas com feedback")
                            #print(f"    {docs_with_feedback} documentos treinados")

                            # Mostra alguns documentos bem avaliados
                            best_docs = self.get_best_documents(3)
                            if best_docs:
                                #print(f"    Melhores documentos:")
                                for doc_id, avg_score in best_docs:
                                    if int(doc_id) < len(self.metadata):
                                        title = self.metadata[int(doc_id)]['title']
                                        #print(f"    - {title}: {avg_score:.1f}/5")

                            #print("  Sistema agora usa inteligência do programa 3!")
                            loaded = True
                            break

                    except Exception as e:
                        print(f"    Erro ao carregar {memory_file}: {e}")
                        continue

            if not loaded:
                print("  Nenhuma memória encontrada - usando busca básica")
                print("  Execute o programa 3 para treinar o sistema!")

        except Exception as e:
            print(f"  Erro ao carregar memória: {e}")
            print("  Usando busca básica")

    def get_best_documents(self, n=5):
        # Retorna os N melhores documentos por avaliação.
        doc_averages = []
        for doc_id, scores in self.document_scores.items():
            if len(scores) >= 1:
                avg_score = sum(scores) / len(scores)
                doc_averages.append((doc_id, avg_score))

        doc_averages.sort(key=lambda x: x[1], reverse=True)
        return doc_averages[:n]

    def calculate_learned_weight(self, chunk_id):
        # Calcula peso baseado no aprendizado do programa 3.
        if not self.has_memory:
            return 1.0  # Peso neutro se não há memória

        chunk_str = str(chunk_id)

        if chunk_str in self.document_scores:
            scores = self.document_scores[chunk_str]
            if len(scores) >= 1:
                # Calcula média das avaliações com peso para feedbacks mais recentes
                total_weight = 0
                total_score = 0

                for i, score in enumerate(scores):
                    weight = 1.0 + (i * 0.1)  # Mais recentes pesam mais
                    total_score += score * weight
                    total_weight += weight

                avg_score = total_score / total_weight

                # Converte para peso (1-5 -> 0.3-2.5)
                if avg_score >= 4.5:
                    return 2.5  # Documentos excelentes
                elif avg_score >= 4.0:
                    return 2.0  # Documentos muito bons
                elif avg_score >= 3.5:
                    return 1.5  # Documentos bons
                elif avg_score >= 2.5:
                    return 1.0  # Documentos neutros
                elif avg_score >= 1.5:
                    return 0.6  # Documentos ruins
                else:
                    return 0.3  # Documentos muito ruins

        return 1.0  # Peso neutro para documentos sem feedback

    def smart_search(self, question, k=5):
        # Busca inteligente usando aprendizado do programa 3.
        try:
            # Cria embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Busca mais documentos para ter opções
            search_k = k * 3 if self.has_memory else k
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
                        if learned_weight > 2.0:
                            confidence = "Excelente"
                        elif learned_weight > 1.3:
                            confidence = "Bom"
                        elif learned_weight < 0.7:
                            confidence = "Baixo"
                        else:
                            confidence = "Médio"
                    else:
                        learned_weight = 1.0
                        adjusted_score = base_score
                        confidence = "Neutro"

                    results.append({
                        'text': self.chunks[idx],
                        'metadata': self.metadata[idx],
                        'base_score': base_score,
                        'learned_weight': learned_weight,
                        'adjusted_score': adjusted_score,
                        'confidence': confidence,
                        'rank': i + 1,
                        'chunk_id': int(idx),
                        'feedback_count': len(self.document_scores.get(str(idx), []))
                    })

            # Ordena por score ajustado se há memória, senão por score base
            if self.has_memory:
                results.sort(key=lambda x: x['adjusted_score'])
            else:
                results.sort(key=lambda x: x['base_score'])

            return results[:k]

        except Exception as e:
            print(f"Erro na busca: {e}")
            return []

    def generate_intelligent_answer(self, question, context_docs):
        # Gera resposta inteligente SEM HuggingFace (evita segmentation fault).
        try:
            if not context_docs:
                return "Desculpe, não encontrei informações relevantes para sua pergunta."

            # Se há memória, ordena por peso aprendido
            if self.has_memory:
                sorted_docs = sorted(context_docs, key=lambda x: x['learned_weight'], reverse=True)
            else:
                sorted_docs = context_docs

            # Identifica palavras-chave da pergunta
            question_lower = question.lower()
            key_words = [word for word in question_lower.split() if len(word) > 3]

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
                
                if self.has_memory:
                    answer = f"Com base na análise inteligente sobre '{question}':\n\n"
                    answer = ''
                else:
                    answer = f"Com base nas informações encontradas sobre '{question}':\n\n"
                    answer = ''

                # Adiciona as informações mais relevantes
                for sentence in best_sentences:
                    if sentence.endswith('.'):
                        answer += f"{sentence} "
                    else:
                        answer += f"{sentence}. "

                # Adiciona estatísticas se há memória
                if self.has_memory:
                    best_doc = sorted_docs[0]
                    if best_doc['feedback_count'] > 0:
                        answer += f"\n\n[Resposta baseada em documento com {best_doc['feedback_count']} avaliações - Confiança: {best_doc['confidence']}]"

                # Adiciona fontes
                if sources:
                    answer += f"\n\nFontes: {', '.join(sources[:3])}\n"

                return answer.strip()
            else:
                # Fallback: resposta simples baseada no melhor documento
                best_text = sorted_docs[0]['text']
                if len(best_text) > 500:
                    best_text = best_text[:500] + "..."

                answer = f"{best_text}"
                # answer = f"Sobre '{question}': {best_text}"

                if sources:
                    answer += f"\n\nFonte: {sources[0]}"

                return answer

        except Exception as e:
            print(f"Erro ao gerar resposta: {e}")
            return f"Erro ao processar sua pergunta: {str(e)}"

    def query(self, question):
        # Executa uma consulta inteligente.
        # print("\n" + "="*60)
        # if self.has_memory:
        #    print("PROCESSANDO COM INTELIGÊNCIA APRENDIDA")
        # else:
        #    print("PROCESSANDO SUA PERGUNTA")
        # print("="*60)
        # print(f"Pergunta: {question}")

        # Busca documentos (inteligente se há memória)
        # if self.has_memory:
        #    print("\nBuscando com inteligência do programa 3...")
        # else:
        #    print("\nBuscando documentos relevantes...")

        docs = self.smart_search(question, k=5)

        if not docs:
            # print("Nenhum documento relevante encontrado.")
            return "Desculpe, não encontrei informações relevantes para sua pergunta."

        # Mostra documentos encontrados (versão compacta)
        # print(f"\nDOCUMENTOS SELECIONADOS:")
        
        for doc in docs[:3]:
            if self.has_memory:
                feedback_info = f" ({doc['feedback_count']} avaliações)" if doc['feedback_count'] > 0 else ""
                #print(f"  {doc['rank']}. {doc['metadata']['title']} - {doc['confidence']}{feedback_info}")
                #print(f"    Peso: {doc['learned_weight']:.2f} | Score: {doc['base_score']:.3f} → {doc['adjusted_score']:.3f}")
            #else:
                #print(f"  {doc['rank']}. {doc['metadata']['title']} - {doc['confidence']}")
                #print(f"    Score: {doc['base_score']:.3f}")
            #print()

        # Gera resposta
        # if self.has_memory:
        #     print("Gerando resposta inteligente otimizada...")
        # else:
        #     print("Gerando resposta...")

        answer = self.generate_intelligent_answer(question, docs)

        return answer

    def interactive_mode(self):
        # Modo interativo inteligente.
        # if self.has_memory:
        #    print("CONSULTA RAG INTELIGENTE ULTRA ROBUSTA")
        #    print("="*50)
        #    print("Sistema usando aprendizado do programa 3!")
        #    print(f"Baseado em {self.total_feedback_queries} consultas com feedback")
        #    print("Respostas otimizadas com base nas suas avaliações")
        # else:
        #    print("CONSULTA RAG BÁSICA ULTRA ROBUSTA")
        #    print("="*50)
        #    print("Sistema usando busca por similaridade")
        #    print("Execute o programa 3 para treinar o sistema!")

        #print("Gerador: ALGORITMO PRÓPRIO (sem HuggingFace)")
        #print("Estabilidade: MÁXIMA (sem segmentation fault)")
        #print("Digite 'sair' para terminar")
        #print("="*50)

        while True:
            try:
                # print("\n" + "="*58)
                question = input("Sua pergunta: ").strip()

                if question.lower() in ['sair', 'exit', 'quit', '']:
                    if self.has_memory:
                        print("\nObrigado por usar o sistema inteligente!")
                        print("Continue treinando com o programa 3 para melhorar ainda mais!")
                    else:
                        print("\nAté logo!")
                        print("Execute o programa 3 para treinar o sistema!")
                    break

                if len(question) < 3:
                    print("Pergunta muito curta. Tente novamente.")
                    continue

                # Processa pergunta
                answer = self.query(question)

                # Mostra resposta
                # if self.has_memory:
                #     print("\nRESPOSTA INTELIGENTE:")
                # else:
                #     print("\nRESPOSTA:")
                #print("-" * 60)
                print(answer)
                #print("-" * 60)

                # if not self.has_memory:
                #     print("\nDica: Execute o programa 3 para treinar o sistema!")
                #     print("   Suas avaliações tornarão as respostas muito melhores!")

            except KeyboardInterrupt:
                print("\n\nPrograma interrompido.")
                break
            except Exception as e:
                print(f"\nErro: {e}")

def main():
    # Função principal.
    # print("RAG QUERY")
    
    try:
        # Inicializa sistema
        rag = UltraRobustRAGQuery()

        # Carrega sistema
        if not rag.load_system():
            print("\nExecute primeiro o treinamento:")
            print("   python 1_rag_trainer_corrigido.py")
            sys.exit(1)

        # Inicia modo interativo
        rag.interactive_mode()

    except KeyboardInterrupt:
        print("\n\nPrograma cancelado.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
