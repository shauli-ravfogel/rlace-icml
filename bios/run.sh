python3 finetune.py --run_id 0 --device 0 --adv 0 --opt sgd &
python3 finetune.py --run_id 1 --device 1 --adv 0 --opt sgd &
python3 finetune.py --run_id 2 --device 2 --adv 0 --opt sgd &
python3 finetune.py --run_id 3 --device 3 --adv 0 --opt sgd
python3 finetune.py --run_id 4 --device 0 --adv 0 --opt sgd &

python3 finetune.py --run_id 0 --device 1 --adv 1 --opt sgd &
python3 finetune.py --run_id 1 --device 2 --adv 1 --opt sgd &
python3 finetune.py --run_id 2 --device 3 --adv 1 --opt sgd
python3 finetune.py --run_id 3 --device 0 --adv 1 --opt sgd &
python3 finetune.py --run_id 4 --device 1 --adv 1 --opt sgd &

python3 finetune.py --run_id 0 --device 2 --adv 1 --mlp_adv 1 --opt sgd &
python3 finetune.py --run_id 1 --device 3 --adv 1 --mlp_adv 1 --opt sgd
python3 finetune.py --run_id 2 --device 0 --adv 1 --mlp_adv 1 --opt sgd &
python3 finetune.py --run_id 3 --device 1 --adv 1 --mlp_adv 1 --opt sgd &
python3 finetune.py --run_id 4 --device 2 --adv 1 --mlp_adv 1 --opt sgd &
