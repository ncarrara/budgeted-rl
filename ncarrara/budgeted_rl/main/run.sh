#!/bin/bash
echo $BASHPID
for i in $(seq $2 $3)
do
     echo "Seed $i"
     python egreedy/main_egreedy.py $1 $i 1
     case $? in
        139)
            echo "erreur 139 on seed $i"
            ;;
        134)
            echo "erreur 134 on seed $i"
     esac
done