#!/usr/bin/env bash

python local_smooth/main.py --config configs/loc_lin_gauss_ker.yml

python local_smooth/main.py --config configs/loc_quadr_gauss_ker.yml

python local_smooth/main.py --config configs/loc_cub_gauss_ker.yml

python local_smooth/main.py --config configs/loc_10_gauss_ker.yml


python local_smooth/main.py --config configs/loc_lin_rect_ker.yml

python local_smooth/main.py --config configs/loc_quadr_rect_ker.yml

python local_smooth/main.py --config configs/loc_cub_rect_ker.yml

python local_smooth/main.py --config configs/loc_10_rect_ker.yml


python local_smooth/main.py --config configs/loc_lin_epan_ker.yml

python local_smooth/main.py --config configs/loc_quadr_epan_ker.yml

python local_smooth/main.py --config configs/loc_cub_epan_ker.yml

python local_smooth/main.py --config configs/loc_10_epan_ker.yml