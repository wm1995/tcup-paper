#!/bin/bash
#PBS -lwalltime=02:00:00
#PBS -lselect=1:ncpus=1:mem=1gb
#PBS -o /rds/general/user/wjm119/home/tcup-paper/run2/hpc-logs/
#PBS -j oe


# Copy files to final directory
echo "=> Copying files at $(date)"
rsync --recursive $PBS_O_WORKDIR/results .

tar -czvf sbc.tar.gz --exclude='checkpoint' results/sbc

cp sbc.tar.gz $PBS_O_WORKDIR

echo "=> Finished run at $(date)"
