PYTHON           = python
SCRIPT           = train_open.py
SCRIPT_GEN       = open_generalized/train_curriculum.py
SCRIPT_CLOSE     = train_close.py
SCRIPT_CLOSE_GEN = close_generalized/train_gen.py
STEPS            = 2000000
STEPS_CLOSE      = 450000
ENVS             = 4

MODEL_PATH           = runs/door_sac/best_model.zip
MODEL_PATH_CLOSE     = runs/door_close_sac/best_model.zip
MODEL_PATH_CLOSE_GEN = runs/door_gen/best_model.zip
LOG_DIR              = runs/tb

.PHONY: train_open_gen train_close_gen play_close_gen train_open play_open train_close play_close board clean help

train_open_gen:
	$(PYTHON) $(SCRIPT_GEN)

train_close_gen:
	$(PYTHON) $(SCRIPT_CLOSE_GEN)

play_close_gen:
	$(PYTHON) $(SCRIPT_CLOSE_GEN) --play --model $(MODEL_PATH_CLOSE_GEN)

help:
	@echo "[INFO: python 3.10 necessario]"
	@echo "[INFO: task open/close]"
	@echo "Comandi:"
	@echo "  make train_<task>     - Avvia l'addestramento del modello"
	@echo "  make play_<task>      - Avvia la valutazione (test) del modello salvato"
	@echo "  make train_<task>_gen - Avvia l'addestramento del modello con generalizzazione"
	@echo "  make play_<task>_gen  - Avvia la valutazione (test) del modello salvato con generalizzazione"
	@echo "  make board            - Avvia TensorBoard per monitorare i log"
	@echo "  make clean            - Rimuove i file temporanei di Python"

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