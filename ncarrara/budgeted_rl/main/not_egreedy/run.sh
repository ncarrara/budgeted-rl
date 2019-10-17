#!/bin/bash
PID=$$
echo $PID
#MESSAGE="[$USER] Processus $PID uses a GPU. If you need a GPU, please kill it with : kill -TERM -$PID."
#curl -X POST -H 'Content-type: application/json' --data  "{\"text\":\"$MESSAGE\"}"  https://hooks.slack.com/services/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
export PYTHONPATH="/home/sequel/ncarrara/phd_code"
N=0
while [ "0"="0" ]
do
  seed=$(($(date +%s%N)/1000000))
#  echo $seed
  seed="$seed"
#  echo $seed
#  u="thisC is a test"
#  var="${u:10:4}"
#  echo "${var}"
  seed=${seed:6:6}
  echo $seed
  python main.py "$1" $seed 1
done