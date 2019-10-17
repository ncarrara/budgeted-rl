#!/bin/bash
echo $BASHPID
for i in 0 1 10 11
do
     echo "Seed $i"
     python egreedy/main_egreedy.py config/slot_filling/compensate.json $i 1
     case $? in
        139)
            echo "erreur 139 on seed $i"
            ;;
        134)
            echo "erreur 134 on seed $i"
     esac
done