PYTHON       = python
SCRIPT       = train_panda_door_sac.py
SCRIPT_CLOSE = train_panda_door_close_sac.py
STEPS        = 2000000
STEPS_CLOSE  = 420000
ENVS         = 2

MODEL_PATH       = runs/door_sac/best_model.zip
MODEL_PATH_CLOSE = runs/door_close_sac/best_model.zip
LOG_DIR          = runs/tb

.PHONY: train_open play_open train_close play_close board clean help

help:
	@echo "[INFO: python 3.10 necessario] Comandi:"
	@echo "  make train   - Avvia l'addestramento del modello"
	@echo "  make play    - Avvia la valutazione (test) del modello salvato"
	@echo "  make board   - Avvia TensorBoard per monitorare i log"
	@echo "  make clean   - Rimuove i file temporanei di Python"

train_open:
	$(PYTHON) $(SCRIPT) --total-steps $(STEPS) --num-envs $(ENVS)

play_open:
	$(PYTHON) $(SCRIPT) --play --model $(MODEL_PATH)

train_close:
	$(PYTHON) $(SCRIPT_CLOSE) --total-steps $(STEPS_CLOSE) --num-envs $(ENVS)

play_close:
	$(PYTHON) $(SCRIPT_CLOSE) --play --model $(MODEL_PATH_CLOSE)

board:
	tensorboard --logdir $(LOG_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +