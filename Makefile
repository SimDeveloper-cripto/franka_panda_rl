PYTHON = python
SCRIPT = train_panda_door_sac.py
STEPS  = 2000000
ENVS   = 2

MODEL_PATH = runs/door_sac/best_model.zip
LOG_DIR    = runs/tb

.PHONY: train play board clean help

help:
	@echo "[INFO: python 3.10 necessario] Comandi:"
	@echo "  make train   - Avvia l'addestramento del modello"
	@echo "  make play    - Avvia la valutazione (test) del modello salvato"
	@echo "  make board   - Avvia TensorBoard per monitorare i log"
	@echo "  make clean   - Rimuove i file temporanei di Python"

train:
	$(PYTHON) $(SCRIPT) --total-steps $(STEPS) --num-envs $(ENVS)

play:
	$(PYTHON) $(SCRIPT) --play --model $(MODEL_PATH)

board:
	tensorboard --logdir $(LOG_DIR)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +