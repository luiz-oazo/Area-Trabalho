Sistema RAG Inteligente com Feedback

Sistema de Retrieval-Augmented Generation (RAG) que aprende com suas avaliações e melhora automaticamente as respostas ao longo do tempo.

Estrutura do Sistema

rag_system/
├── 1_rag_trainer.py                              # Treiner inicial
├── 1_rag_trainer_universal.py                    # Trainer universal
├── 2_rag_query.py                                # Consultas
├── 3_rag_feedback.py                             # Treiner Feedback 
└── rag_system/                                   # Dados do sistema
    ├── documents.json                            # Documentos coletados
    ├── rag_data.pkl                              # Chunks e metadados
    ├── faiss_index.bin                           # Índice de busca
    ├── embedding_model/                          # Modelo de embeddings
    ├── feedback_memory.json                      # Memória principal
    └── feedback_backup.json                      # Backup de segurança


Treinamento (wiki)

python 1_rag_trainer.py

Treinamento mais de um tipo de fonte

python 1_rag_trainer_universal.py

Consultas Básicas

python 2_rag_query.py

Treinamento com Feedback

python 3_rag_feedback.py

Como o Sistema Aprende

5 → Peso 2.5 (prioridade máxima)
4 → Peso 2.0 (alta prioridade)
3 → Peso 1.0 (neutro)
2 → Peso 0.6 (baixa prioridade)
1 → Peso 0.3 (evitado)

Memória Persistente

- Todas as avaliações são salvas automaticamente
- Sistema carrega memória na próxima execução
- Múltiplos backups para segurança máxima
- Compatível com todas as versões
