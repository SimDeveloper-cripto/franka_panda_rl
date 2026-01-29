# Franka Panda RL - Progetto Manipolazione Porta 🦾🚪

> **Progetto Universitario** - Reinforcement Learning per Manipolazione Robotica con Franka Emika Panda.
>
> 🇮🇹 **Documentazione Ufficiale**

---

# Documentazione Completa del Progetto

Questo repository contiene l'implementazione di un agente di **Reinforcement Learning (RL)** progettato per risolvere task di manipolazione robotica complessa utilizzando un manipolatore **Franka Emika Panda** nell'ambiente di simulazione **Robosuite**.

L'obiettivo è addestrare una rete neurale in grado di **Aprire** e **Chiudere** una porta, gestendo non solo la cinematica del robot ma anche le interazioni fisiche (contatti, attriti) e dimostrando capacità di **Generalizzazione** e robustezza.

---

## 🤖 Algoritmo: Soft Actor-Critic (SAC)

Il core dell'apprendimento è basato su **SAC (Soft Actor-Critic)**, un algoritmo state-of-the-art di tipo **Off-Policy** per il controllo continuo.

**Cosa significa Off-Policy?**
A differenza degli algoritmi On-Policy (che imparano solo dai dati raccolti dalla policy corrente), SAC utilizza un **Replay Buffer** per immagazzinare le esperienze passate $(s, a, r, s')$. L'agente può quindi imparare da dati raccolti in momenti precedenti (o anche da dimostrazioni esterne), rendendo l'algoritmo estremamente efficiente nell'uso dei campioni (*Sample Efficient*) e ideale per la robotica.

**La Massimizzazione dell'Entropia**
SAC non cerca solo di massimizzare la ricompensa cumulativa, ma anche l'entropia della policy:

$$J(\pi) = \sum_{t} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} [r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$$

Dove:
*   $r(s_t, a_t)$: Ricompensa ottenuta dal task.
*   $\mathcal{H}(\pi(\cdot|s_t))$: Entropia della policy. Un'entropia alta incentiva l'esplorazione e impedisce la convergenza prematura su ottimi locali.
*   $\alpha$: Temperatura. Bilancia automaticamente il trade-off tra esplorazione (inizio training) e sfruttamento (fine training).

---

## 🏗️ Architettura Software Dettagliata

Il progetto segue un'architettura modulare per garantire riproducibilità e separazione delle responsabilità.

### 1. Configurazione Centralizzata (`TrainConfig`)
Tutti gli iperparametri sono definiti in **Dataclasses** Python (es. `config/train_open_config.py`).
*   Elimina i *magic number* dal codice.
*   Oggetto `cfg`: Iniettato nell'ambiente, controlla pesi della reward (`w_progress`, `w_delta`), parametri fisici (`horizon`) e di apprendimento.

### 2. Gerarchia e Wrapper degli Ambienti
L'ambiente di simulazione è costruito a strati:
1.  **Backend (Robosuite/MuJoCo)**: Risolve le equazioni differenziali della dinamica *multi-body* e gestisce le collisioni.
2.  **Gymnasium Adapter**: Converte gli spazi di input/output proprietari in standard `gym.spaces`. Esegue il *flattening* delle osservazioni multimodali (propriocezione + stato oggetto) in un tensore 1D per la rete neurale.
3.  **Task-Specific Wrappers (`GeneralizedDoorEnv`)**: Estendono la classe base iniettando logiche avanzate come la *Domain Randomization* nel metodo `reset()`.

---

## 🔬 Analisi Scientifica della Generalizzazione

La sfida principale in robotica non è risolvere un task, ma risolverlo in condizioni variabili (**Sim-to-Real Gap**). Questo progetto implementa due strategie distinte basate sulla teoria del controllo robusto.

### 1. Dynamics Randomization: Apprendimento di Impedenza (Apertura)
Nel task di apertura, l'agente deve affrontare porte con proprietà fisiche ignote.
*   **Il Problema**: Una policy "rigida" (che impara solo una traiettoria di posizione $q(t)$) fallisce se l'attrito della porta cambia. Se la porta è più pesante del previsto, l'errore di inseguimento aumenta e il robot si blocca.
*   **La Soluzione (Curriculum a Stadi)**:
    *   Durante il training, variamo i parametri dinamici della cerniera: **Attrito Viscoso** ($\mu$) e **Smorzamento** ($k$).
    *   L'agente osserva lo stato $s_t$ (angolo porta) e l'errore rispetto all'apertura attesa.
    *   Per massimizzare la reward in stadi ad alto attrito, la rete apprende implicitamente un controllo di **impedenza variabile**: modula la forza/torque applicata in risposta alla resistenza percepita.
    *   **Risultato**: La policy $\pi(a|s)$ non è una semplice riproduzione di movimenti, ma un controllore attivo che spinge "più forte" se la porta non si apre come previsto.

### 2. Spatial Domain Randomization: Reiezione dei Disturbi (Chiusura)
Nel task di chiusura, l'incertezza è geometrica.
*   **Il Problema**: Errori di calibrazione tra base del robot e porta. Una policy *open-loop* (cieca) mancherebbe la maniglia se la porta fosse spostata anche solo di 1cm.
*   **La Soluzione (Curriculum Adattivo)**:
    *   Modelliamo la posizione o orientamento della porta come variabili aleatorie:
        $$P_{door} \sim \mathcal{N}(P_{nom}, \sigma_{curriculum})$$
        Dove:
        *   $P_{door}$: Posizione/Orientamento effettivo della porta nell'episodio corrente.
        *   $\mathcal{N}$: Indica una Distribuzione Normale (Gaussiana).
        *   $P_{nom}$: Posizione nominale standard della porta (il "centro" ideale).
        *   $\sigma_{curriculum}$: Deviazione standard corrente. È il parametro che controlla *quanto* la porta può essere spostata. Inizia da 0 (nessun disturbo) e cresce man mano che l'agente diventa bravo.
    *   Il parametro $\sigma$ (intensità del disturbo) cresce progressivamente solo quando l'agente ha un Success Rate > 85%.
*   **Analisi del Controllo**:
    *   Questo costringe l'agente a ignorare le coordinate assolute e a focalizzarsi sulle **coornidate relative** (End-Effector vs Maniglia).
    *   La rete neurale diventa un **regolatore a ciclo chiuso** (Closed-Loop Controller): impara a correggere l'errore di posizionamento in tempo reale. Se la maniglia si sposta, l'agente adatta la traiettoria "al volo" per intercettarla. È una forma di *Visual Servoing* (o meglio, State Servoing) appreso.

---

## 🧠 Analisi Approfondita del Reward Shaping

La funzione di ricompensa è il segnale che guida l'ottimizzazione. È progettata per evitare minimi locali e comportamenti instabili.

### Task Apertura (Fasi Sequenziali)
La reward cambia dinamicamente in base allo stato del task:
1.  **Fase di Manipolazione**:
    *   `w_progress`: Premia la riduzione della distanza angolare (Componente proporzionale).
    *   `w_delta`: Premia la velocità positiva (Componente derivativa, serve a vincere l'attrito di primo distacco).
2.  **Fase di Post-Successo (Return Stage)**:
    *   Una volta aperta la porta, vogliamo evitare che il robot collassi o rimanga in tensione.
    *   Attiviamo un attrattore verso la posizione iniziale (`w_return_pos`) e una penalità se la porta si richiude (`w_door_regress`).
    *   L'agente impara così una manovra pulita: **Afferra $\rightarrow$ Apre $\rightarrow$ Rilascia/Indietreggia**.

### Task Chiusura (Stabilizzazione)
Qui l'obiettivo critico è la precisione finale.
*   **Bonus "Freezing"**: Nel reward shaping della classe `GeneralizedDoorEnv`, abbiamo inserito termini specifici per quando la porta è chiusa (`is_closed=True`).
    *   `reward -= 0.5 * ||action||`: Penalità su *qualsiasi* torque applicata.
    *   `reward += 0.5` **se** `||velocity|| < 0.1`: Bonus esplicito per la velocità nulla.
*   **Effetto**: Questo insegna all'agente a "spegnersi" o mantenere una posa rigida una volta completato il lavoro, prevenendo oscillazioni o riaperture accidentali.

---

## 🛠️ Tecniche di Apprendimento

### Action Smoothing (Simulazione Inerzia)
Per ridurre il gap con i robot reali (che hanno limiti di banda), le azioni della rete neurale ($a_{NN}$) vengono filtrate:
$$a_{motori}^{(t)} = \alpha \cdot a_{motori}^{(t-1)} + (1-\alpha) \cdot a_{NN}^{(t)}$$
Dove:
*   $a_{motori}^{(t)}$: Azione finale inviata ai motori del robot al passo temporale corrente $t$.
*   $a_{motori}^{(t-1)}$: Azione inviata ai motori al passo precedente (memoria inerziale).
*   $a_{NN}^{(t)}$: Azione "grezza" appena calcolata dalla Rete Neurale (Policy).
*   $\alpha$: Coefficiente di smoothing (es. 0.8). Determina quanto pesa la "storia" rispetto al nuovo comando. Un $\alpha$ alto rende il movimento molto fluido ma più lento a reagire; un $\alpha$ basso lo rende più reattivo ma "scattoso".
Questo funge da filtro passa-basso, rendendo i movimenti fluidi e meno aggressivi (less jerk), migliorando il transfer sulla fisica reale e la convergenza.

---

## 🚀 Setup e Comandi

### Prerequisiti Windows (Fix Applicati)
Il progetto include patch automatiche per l'ecosistema Windows:
*   **DLL Injection**: Gli script iniettano il path `site-packages\mujoco` nelle DLL directory di sistema a runtime.
*   **Compatibilità Numpy**: Verifica automatica/compatibilità con Numpy `2.x`.

### Esecuzione Simulazioni

Da terminale (con virtual environment attivo):

**1. Apertura (Generalized / Curriculum)**
```powershell
& .venv\Scripts\python.exe open_generalized/play.py
```
*Visualizza l'agente che affronta stadi di difficoltà fisica crescente.*

**2. Chiusura (Generalized / Random Position)**
```powershell
# Esegue il play con Curriculum Level = 1.0 (Massima Randomizzazione)
& .venv\Scripts\python.exe close_generalized/train_gen.py --play
```

**3. Training da Zero**
*   `make train_open_gen`: Avvia training con curriculum a stadi.
*   `make train_close_gen`: Avvia training con randomizzazione geometrica progressiva.
