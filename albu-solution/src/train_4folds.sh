tmux set-option remain-on-exit on
tmux split-window -t 0 -p 50 './train.sh 1'

sleep 20
tmux split-window -t 0 -p 50 './train.sh 2'

sleep 20
tmux split-window -t 1 -p 50 './train.sh 3'

sleep 20
./train.sh 0
