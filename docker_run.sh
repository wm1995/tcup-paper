docker run --mount type=bind,source=$(pwd)/data,target=/mcmc/data --mount type=bind,source=$(pwd)/results,target=/mcmc/results linmix --quiet -r 0 data/kelly.json results/linmix-kelly.json
