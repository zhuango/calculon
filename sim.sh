python3 ./bin/calculon \
    llm ./models/llama2-7b.json \
    ./examples/3072_t1_p1_d1_mbs1_full.json \
    ./systems/a100_80e.json \
    ./calculon_stats.json \
    -p ./calculon_peers.json \
    --layers