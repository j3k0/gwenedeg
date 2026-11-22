llama.cpp/build/bin/llama-server -m {model} -c 98304 -np 1 -t 14 -fa on -ctk q8_0 -ctv q4_0 -ub 512 -b 2048 -lm none --no-repack --ctx-checkpoints 4 -cram 512 --jinja --host 127.0.0.1 --port {port}
