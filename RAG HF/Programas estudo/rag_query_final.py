import pickle
import os
import faiss
import numpy as np
import json
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import Dataset
from sentence_transformers import SentenceTransformer

class SimpleRAGQuery:
    def __init__(self, save_dir="rag_system"):
        self.save_dir = save_dir
        self.tokenizer = None
        self.rag_model = None
        self.dataset = None
        self.index = None
        self.embedding_model = None

    def load_rag_system(self):
        """Carrega todos os componentes do sistema RAG."""
        try:
            print("🔄 Carregando sistema RAG...")

            # Verificar se diretório existe
            if not os.path.exists(self.save_dir):
                print(f"❌ Diretório {self.save_dir} não encontrado!")
                print("Execute primeiro o 'rag_trainer_fixed.py' para treinar o sistema.")
                return False

            # Carregar tokenizer
            print("  Carregando tokenizer...")
            self.tokenizer = RagTokenizer.from_pretrained(f"{self.save_dir}/tokenizer")

            # Carregar dataset
            print("  Carregando dataset...")
            try:
                self.dataset = Dataset.load_from_disk(f"{self.save_dir}/dataset")
            except Exception as e:
                print(f"    Erro ao carregar dataset do disco: {e}")
                print("    Tentando carregar do arquivo pickle...")
                with open(f"{self.save_dir}/dataset.pkl", "rb") as f:
                    dataset_dict = pickle.load(f)
                    self.dataset = Dataset.from_dict(dataset_dict)

            # Carregar índice Faiss
            print("  Carregando índice Faiss...")
            self.index = faiss.read_index(f"{self.save_dir}/faiss_index.bin")

            # Carregar modelo de embedding
            print("  Carregando modelo de embedding...")
            self.embedding_model = SentenceTransformer(f"{self.save_dir}/embedding_model")

            # Recriar o índice Faiss no dataset
            print("  Recriando índice Faiss no dataset...")
            if 'embeddings' in self.dataset.column_names:
                # Adicionar índice Faiss ao dataset
                self.dataset.add_faiss_index(column='embeddings', custom_index=self.index)

            # Criar retriever com o dataset indexado
            print("  Criando retriever...")
            retriever = RagRetriever.from_pretrained(
                "facebook/rag-sequence-nq", 
                indexed_dataset=self.dataset
            )

            # Carregar modelo RAG completo
            print("  Carregando modelo RAG...")
            self.rag_model = RagSequenceForGeneration.from_pretrained(
                f"{self.save_dir}/model"
            )

            # Atualizar o retriever do modelo
            self.rag_model.retriever = retriever

            print("✅ Sistema RAG carregado com sucesso!")
            return True

        except Exception as e:
            print(f"❌ Erro ao carregar sistema RAG: {e}")
            print("\nDetalhes do erro:")
            import traceback
            traceback.print_exc()
            print("\nExecute primeiro o 'rag_trainer_fixed.py' para treinar o sistema.")
            return False

    def query(self, question):
        """Faz uma pergunta ao sistema RAG e retorna a resposta."""
        try:
            print("🔍 Processando pergunta...")

            # Tokenizar a pergunta
            inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)

            # Gerar resposta
            with torch.no_grad():
                generated = self.rag_model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            # Decodificar resposta
            answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

            # Limpar a resposta (remover a pergunta se estiver incluída)
            if question.lower() in answer.lower():
                answer = answer.replace(question, "").strip()

            return answer

        except Exception as e:
            return f"❌ Erro ao processar pergunta: {e}"

    def get_retrieved_documents(self, question, k=3):
        """Retorna os documentos mais relevantes para a pergunta."""
        try:
            # Criar embedding da pergunta
            question_embedding = self.embedding_model.encode([question])
            question_embedding = np.array(question_embedding).astype('float32')

            # Buscar documentos similares
            scores, indices = self.index.search(question_embedding, k)

            retrieved_docs = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.dataset) and idx >= 0:
                    doc = {
                        'title': self.dataset[int(idx)]['title'],
                        'text': self.dataset[int(idx)]['text'][:300] + "...",  # Primeiros 300 chars
                        'score': float(scores[0][i]),
                        'url': self.dataset[int(idx)]['url']
                    }
                    retrieved_docs.append(doc)

            return retrieved_docs

        except Exception as e:
            print(f"❌ Erro ao recuperar documentos: {e}")
            return []

    def simple_query_without_rag(self, question):
        """Faz uma pergunta simples sem usar o retriever (para teste)."""
        try:
            # Usar apenas o generator do modelo RAG
            inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=512)

            # Gerar resposta usando apenas o generator
            with torch.no_grad():
                generated = self.rag_model.generator.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=100,
                    num_beams=2,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return answer

        except Exception as e:
            return f"❌ Erro no modo simples: {e}"

def main():
    """Função principal com prompt para pergunta."""
    # Importar torch aqui para evitar problemas de importação
    try:
        import torch
        globals()['torch'] = torch
    except ImportError:
        print("❌ PyTorch não encontrado. Instale com: pip install torch")
        return

    rag = SimpleRAGQuery()

    # Carregar sistema RAG
    if not rag.load_rag_system():
        return

    print("\n" + "="*60)
    print("🤖 SISTEMA RAG - CONSULTA SIMPLES")
    print("="*60)
    print("Digite sua pergunta abaixo:")
    print("-"*60)

    # Prompt para pergunta
    question = input("\n❓ Sua pergunta: ").strip()

    if not question:
        print("❌ Nenhuma pergunta foi digitada.")
        return

    print("\n" + "-"*60)

    # Mostrar documentos recuperados
    print("📚 DOCUMENTOS RELEVANTES:")
    print("-"*30)
    retrieved_docs = rag.get_retrieved_documents(question)
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs, 1):
            print(f"{i}. {doc['title']} (Score: {doc['score']:.4f})")
            print(f"   {doc['text']}")
            print(f"   URL: {doc['url']}")
            print()
    else:
        print("Nenhum documento relevante encontrado.")

    # Processar pergunta
    print("🤖 GERANDO RESPOSTA...")
    answer = rag.query(question)

    # Mostrar resultado
    print(f"\n🤖 RESPOSTA:")
    print("-"*60)
    print(answer)
    print("\n" + "="*60)
    print("✅ Consulta finalizada!")

if __name__ == "__main__":
    main()
