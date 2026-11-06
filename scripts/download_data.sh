#!/usr/bin/env bash
# Download the raw data (about 100 MB) into data/raw, at the versions used in the paper.
#   qdgp (Saarinen et al., 2024): GMB, DisGeNET and Open Targets disease-gene sets.
#   RWR-MH (Valdeolivas et al., 2019): MX-V layers, OMIM disease network, gene-phenotype links.
set -euo pipefail
raw=${1:-data/raw}
mkdir -p "$raw"
cd "$raw"

fetch() {
    local url=$1 dir=$2 commit=$3
    [ -d "$dir" ] || git clone -q "$url" "$dir"
    git -C "$dir" checkout -q "$commit"
}

fetch https://github.com/markgolds/qdgp.git qdgp 5566e17d40a2e7dd76020391bb325464f6cb5bc7
fetch https://github.com/alberto-valdeolivas/RWR-MH.git RWR-MH ee72a71b4a757a21d81a7d34af01c21f84e0abf0
unzip -qo RWR-MH/RWR-MH.zip -d rwrmh
unzip -qo RWR-MH/RWR-M.zip -d rwrm
echo "raw data in $(pwd)"
