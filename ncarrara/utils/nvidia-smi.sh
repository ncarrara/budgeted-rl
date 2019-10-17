#!/bin/sh
NVIDIA=$(nvidia-smi)

NVIDIA=$(echo "$NVIDIA"|sed -n -e '/|=\{1,\}|/,$p')
NVIDIA=$(echo "$NVIDIA"|sed 's/ \{1,\}/ /g')
NVIDIA=$(echo "$NVIDIA"|sed 's/|//g')
NVIDIA=$(echo "$NVIDIA"|sed 's/=//g')
NVIDIA=$(echo "$NVIDIA"|sed 's/-//g')
NVIDIA=$(echo "$NVIDIA"|sed 's/\+//g')
DATA=$(date "+%H:%M:%S   %d/%m/%y")
MESSAGE="\n---------------------------------\nGPU status ($DATA) $NVIDIA\n---------------------------------\n"
#MESSAGE="test"
#echo "$MESSAGE"
curl -X POST -H 'Content-type: application/json' --data  "{\"text\":\"$MESSAGE\"}"  https://hooks.slack.com/services/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