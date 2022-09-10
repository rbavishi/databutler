python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v0" train_embeddings --training-data-type="v0"
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v1" train_embeddings --training-data-type="v1"
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v2" train_embeddings --training-data-type="v2"
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v3" train_embeddings --training-data-type="v3"
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v4" train_embeddings --training-data-type="v1" --per-equiv-class=20
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v5" train_embeddings --training-data-type="v2" --per-equiv-class=10
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/model_v6" train_embeddings --training-data-type="v3" --per-equiv-class=10

python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/crossencoder_v1" --cross-encoder-path="./scratch/pandas_mining_06_22/crossencoder_v1" train_crossencoder --training-data-type="v1"
python databutler/mining/static_pandas_mining/autodoc_search.py EmbeddingBasedSearcher --campaign-dir="./scratch/pandas_mining_06_22" --model-path="./scratch/pandas_mining_06_22/crossencoder_v2" --cross-encoder-path="./scratch/pandas_mining_06_22/crossencoder_v2" train_crossencoder --training-data-type="v2"
