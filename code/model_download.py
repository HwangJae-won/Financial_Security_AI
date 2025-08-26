from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LGAI-EXAONE/EXAONE-Deep-7.8B",
    revision="17b70148e344c28f54a542a030a805b8c96be8c3",
    local_dir="/workspace/models/EXAONE-Deep-7.8B",
    local_dir_use_symlinks=False,
    resume_download=True,
)

snapshot_download(
    repo_id="intfloat/multilingual-e5-small",
    revision="c007d7ef6fd86656326059b28395a7a03a7c5846",
    local_dir="/workspace/models/multilingual-e5-small",
    local_dir_use_symlinks=False,
    resume_download=True,
)

snapshot_download(
    repo_id="Alibaba-NLP/gte-multilingual-reranker-base",
    revision="8215cf04918ba6f7b6a62bb44238ce2953d8831c",
    local_dir="/workspace/models/gte-multilingual-reranker-base",
    local_dir_use_symlinks=False,
    resume_download=True,
)
