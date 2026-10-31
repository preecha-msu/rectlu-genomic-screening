@echo off
echo Running Unified Experimental Pipeline (E1-E6)...
python experiments/step01_monte_carlo_moments.py
python experiments/step02_exp1_motif_and_did.py
python experiments/step03_exp2_4_mechanistic.py
python experiments/step04_exp3_lw_grid.py
python experiments/step05_exp5_6_hybrid_loss.py
python experiments/step06_cross_domain.py
python experiments/step07_real_giab_eval.py
echo All experiments completed!
pause
