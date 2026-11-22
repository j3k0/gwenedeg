llama.cpp/build/bin/llama-server -m {model} -c 98304 -np 4 -t 14 -fa on -ctk f16 -ctv f16 -ub 512 -b 2048 -lm none --no-repack --ctx-checkpoints 2 -cram 512 --jinja --host 127.0.0.1 --port {port}
