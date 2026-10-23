# CodeMaster

**Agente AI per l'analisi intelligente di repository di codice**

CodeMaster usa **RAG + ReAct** per aiutarti a comprendere, analizzare e migliorare qualsiasi codebase, che sia grande, non documentata o generata da AI. Analizza architettura, spiega funzionalità, trova bug e suggerisce miglioramenti.

---

## 📌 Situazione

Repository di grandi dimensioni, codice senza documentazione o generato da AI: **CodeMaster** risolve questi problemi offrendo:
- Analisi automatica della struttura
- Spiegazioni teoriche e pratiche
- Identificazione di bug e code smell
- Suggerimenti architetturali

---

## ✅ Installazione

```bash
# 1. Clona o scarica il progetto
cd /percorso/destinazione
git clone https://github.com/tuo-username/CodeMaster.git
cd CodeMaster

# 2. Crea ambiente virtuale
python -m venv .venv

# 3. Attivalo
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 4. Installa dipendenze
pip install -r requirements.txt

# 5. Configura API Key (copia .env.example in .env)
GEMINI_API_KEY=tua_chiave
GROQ_API_KEY=tua_chiave
CLOUD_API_KEY=tua_chiave
```

> ⚠️ **Nota**: `providers.json` è modificabile in base alle tue API o aggiornamenti dei modelli.

---

## 🚀 Esecuzione

### Modalità CLI (Terminale/IDE)

```bash
# Repository locale
python run_cli.py /percorso/alla/repo

# Repository Git remota
python run_cli.py https://github.com/utente/repo.git

# Opzioni principali
--provider gemini_3.8    # Provider principale (default: gemini_3.8 se disponibile)
--fallback groq,gemini   # Provider di riserva in caso di rate limit
--model gemini-3.8-flash # Modello specifico
--embedder auto          # Strategia embedding: auto, sentence_transformers, gemini_api, hashing
--rebuild                # Forza ricostruzione indice
--allow-exec             # Abilita esecuzione comandi Python (su copia temp)
--allow-web              # Abilita ricerca web
--no-cache               # Disabilita cache risposte
--obs-budget 50000       # Budget memoria output
```

### Modalità Web (Gradio)

```bash
# Avvia server web
python run_chat.py /percorso/repo --port 7860 --share

# Accesso: http://localhost:7860
```

---

## 🎮 Comandi Rapidi (solo CLI - 0 token API)

| Comando | Descrizione |
|---------|-------------|
| `/map` | Mappa generale repository |
| `/onboarding` | Guida iniziale: file chiave, dove partire |
| `/outline <file>` | Struttura AST di un file |
| `/symbol <nome>` | Trova definizione funzione/classe |
| `/teoria` | Modalità analisi teorica |
| `/pratica` | Modalità analisi pratica |
| `/bug` | Modalità bug hunting |
| `/design` | Modalità design review |
| `/auto` | Modalità automatica |
| `/clear` | Pulisce memoria conversazione |
| `/exit` | Esci |

---

## 📊 Modalità di Analisi

### Teorica (`/teoria`)
Spiega **come** e **perché** il codice funziona: pattern, design decision, principi.

### Pratica (`/pratica`)
Mostra **esempi concreti**: flussi di esecuzione, relazioni tra componenti.

### Bug (`/bug`)
Identifica **problemi**: bug, anti-pattern, code smell, security issue.

### Design (`/design`)
Suggerisce **miglioramenti**: refactoring, ottimizzazioni, principi SOLID.

> 💡 **Auto**: CodeMaster seleziona automaticamente la modalità in base alla domanda.

---

## 🛠️ Configurazione

### File Importanti

| File | Descrizione |
|------|-------------|
| `.env` | API Key (NON committare!) |
| `providers.json` | Configurazione provider/modelli (modificabile) |
| `requirements.txt` | Dipendenze Python |

### Personalizzazione

1. **Aggiungere provider** in `providers.json`:
```json
{
  "mio_provider": {
    "model_id": "nome-modello",
    "env_key": "MIA_API_KEY",
    "api_base": "https://api.endpoint.com"
  }
}
```

2. **Aggiungere API Key** in `.env`:
```env
MIA_API_KEY=tua_chiave
```