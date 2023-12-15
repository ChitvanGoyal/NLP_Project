# NLP_Project

This project assesses BERT-based passage retrieval systems, addressing scalability
challenges for these systems by implementing two-stage retrieval architecture. Fast
probabilistic methods are used for initial filtering followed by re-ranking with
BERT. Our findings show language models enhance retrieval quality, even at small
re-ranking candidate set sizes. Four different architectures were assessed; notably,
query expansion with DocT5query ahead of initial filtering significantly improves
re-ranking performance in small candidate sets with negligible latency impact.
ColBERT exhibits significantly improves query latency with a modest reduction
in re-ranking quality. Our results highlight trade-offs between retrieval quality
and latency, providing insights suitable uses cases for the evaluated re-ranking
strategies architectures.


