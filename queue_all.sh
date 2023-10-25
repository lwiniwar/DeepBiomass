#!/usr/bin/bash

#bash run.slrm 1e-6 1e-8 8192 8
#bash run.slrm 2e-6 1e-8 8192 8
#bash run.slrm 3e-6 1e-8 8192 8
#bash run.slrm 4e-6 1e-8 8192 8
#bash run.slrm 5e-6 1e-8 8192 8

bash run_stage2.slrm 1e-4 1e-8
bash run_stage2.slrm 1e-5 1e-8
bash run_stage2.slrm 1e-6 1e-8
bash run_stage2.slrm 1e-7 1e-8


#bash run.slrm 1e-3 1 8192 16
#bash run.slrm 1e-4 1 8192 16
#bash run.slrm 1e-5 1 8192 16
#bash run.slrm 1e-6 1 8192 16
#
#
#bash run.slrm 1e-3 1 16384 16
#bash run.slrm 1e-4 1 16384 16
#bash run.slrm 1e-5 1 16384 16
#bash run.slrm 1e-6 1 16384 16
#
#
#bash run.slrm 1e-3 1 4096 32
#bash run.slrm 1e-4 1 4096 32
#bash run.slrm 1e-5 1 4096 32
#bash run.slrm 1e-6 1 4096 32
#
#
#bash run.slrm 1e-3 1 4096 8
#bash run.slrm 1e-4 1 4096 8
#bash run.slrm 1e-5 1 4096 8
#bash run.slrm 1e-6 1 4096 8
