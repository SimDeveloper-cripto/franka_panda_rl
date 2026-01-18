PYTHON      		 = python
STEPS       		 = 2000000
STEPS_CLOSE 		 = 450000
ENVS        		 = 4
LOG_DIR     		 = runs/tb
SCRIPT               = train_open.py
SCRIPT_GEN           = open_generalized/train_curriculum.py
MODEL_PATH           = runs/open_det/best_model.zip
SCRIPT_CLOSE         = train_close.py
SCRIPT_CLOSE_GEN     = close_generalized/train_gen.py
MODEL_PATH_CLOSE     = runs/close_det/best_model.zip
MODEL_PATH_CLOSE_GEN = runs/close_gen/best_model.zip

.PHONY: train_open_gen train_close_gen play_close_gen train_open play_open train_close play_close board help

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