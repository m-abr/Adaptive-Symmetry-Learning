#!/bin/bash

if [ "$#" -ne 2 ]; then 
    echo "Usage: ./ant_batch.sh model_name params_folder"
    echo "E.g.:  ./ant_batch.sh my_new_model parameters/A1.1/ASL"
    exit
fi

x=12
echo "Launching $x * $1 in 2s..."

sleep 2
trap '' INT # ignore sigint

for (( i=0; i<$x; i++ )); do
echo "python3 symm_ant.py $1 $2 $i"
python3 symm_ant.py $1 $2 $i &
done

wait () ( # send ctrl+c to kill everything
   trap - INT
   sleep infinity
)

wait
pkill -e -9 -f "python3 symm_ant.py $1 $2"
echo FINISHED!
