python3 run_bios.py --device -1 --run_id 0 --do_inlp 1 --do_rlace 0 --finetune_mode no-adv &
python3 run_bios.py --device -1 --run_id 1 --do_inlp 1 --do_rlace 0 --finetune_mode no-adv &
python3 run_bios.py --device -1 --run_id 2 --do_inlp 1 --do_rlace 0 --finetune_mode no-adv &
python3 run_bios.py --device -1 --run_id 3 --do_inlp 1 --do_rlace 0 --finetune_mode no-adv &
python3 run_bios.py --device -1 --run_id 4 --do_inlp 1 --do_rlace 0 --finetune_mode no-adv &

python3 run_bios.py --device -1 --run_id 0 --do_inlp 1 --do_rlace 0 --finetune_mode linear-adv &
python3 run_bios.py --device -1 --run_id 1 --do_inlp 1 --do_rlace 0 --finetune_mode linear-adv &
python3 run_bios.py --device -1 --run_id 2 --do_inlp 1 --do_rlace 0 --finetune_mode linear-adv &
python3 run_bios.py --device -1 --run_id 3 --do_inlp 1 --do_rlace 0 --finetune_mode linear-adv &
python3 run_bios.py --device -1 --run_id 4 --do_inlp 1 --do_rlace 0 --finetune_mode linear-adv &

python3 run_bios.py --device -1 --run_id 0 --do_inlp 1 --do_rlace 0 --finetune_mode mlp-adv &
python3 run_bios.py --device -1 --run_id 1 --do_inlp 1 --do_rlace 0 --finetune_mode mlp-adv &
python3 run_bios.py --device -1 --run_id 2 --do_inlp 1 --do_rlace 0 --finetune_mode mlp-adv &
python3 run_bios.py --device -1 --run_id 3 --do_inlp 1 --do_rlace 0 --finetune_mode mlp-adv &
python3 run_bios.py --device -1 --run_id 4 --do_inlp 1 --do_rlace 0 --finetune_mode mlp-adv &

python3 run_bios.py --device 3 --run_id 0 --do_inlp 0 --do_rlace 1 --finetune_mode no-adv &
python3 run_bios.py --device 0 --run_id 1 --do_inlp 0 --do_rlace 1 --finetune_mode no-adv &
python3 run_bios.py --device 1 --run_id 2 --do_inlp 0 --do_rlace 1 --finetune_mode no-adv
python3 run_bios.py --device 2 --run_id 3 --do_inlp 0 --do_rlace 1 --finetune_mode no-adv &
python3 run_bios.py --device 3 --run_id 4 --do_inlp 0 --do_rlace 1 --finetune_mode no-adv &

python3 run_bios.py --device 0 --run_id 0 --do_inlp 0 --do_rlace 1 --finetune_mode linear-adv &
python3 run_bios.py --device 1 --run_id 1 --do_inlp 0 --do_rlace 1 --finetune_mode linear-adv &
python3 run_bios.py --device 2 --run_id 2 --do_inlp 0 --do_rlace 1 --finetune_mode linear-adv
python3 run_bios.py --device 3 --run_id 3 --do_inlp 0 --do_rlace 1 --finetune_mode linear-adv &
python3 run_bios.py --device 0 --run_id 4 --do_inlp 0 --do_rlace 1 --finetune_mode linear-adv &

python3 run_bios.py --device 1 --run_id 0 --do_inlp 0 --do_rlace 1 --finetune_mode mlp-adv &
python3 run_bios.py --device 2 --run_id 1 --do_inlp 0 --do_rlace 1 --finetune_mode mlp-adv
python3 run_bios.py --device 3 --run_id 2 --do_inlp 0 --do_rlace 1 --finetune_mode mlp-adv &
python3 run_bios.py --device 0 --run_id 3 --do_inlp 0 --do_rlace 1 --finetune_mode mlp-adv &
python3 run_bios.py --device 1 --run_id 4 --do_inlp 0 --do_rlace 1 --finetune_mode mlp-adv &