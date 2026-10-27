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

### CLI LLM autenticati localmente

`--provider` accetta i nomi configurati `claude`, `codex`, `vibe`, `agy` e anche gli alias `claude_cli`, `codex_cli`, `mistral_vibe_cli`, `antigravity_cli`. Installa la CLI del provider e completa il login nel terminale prima di avviare CodeMaster:

| Alias | Eseguibile | Installazione e login |
|---|---|---|
| `claude_cli` | `claude` | [Claude Code](https://docs.anthropic.com/en/docs/claude-code/cli-usage) |
| `codex_cli` | `codex` | [Codex CLI](https://developers.openai.com/codex/cli/) |
| `mistral_vibe_cli` | `vibe` | [Mistral Vibe CLI](https://docs.mistral.ai/vibe/code/cli/install-setup) |
| `antigravity_cli` | `agy` | [Antigravity CLI](https://github.com/google-antigravity/antigravity-cli) |

```bash
python run_cli.py /percorso/repo --provider claude_cli
python run_cli.py /percorso/repo --provider codex_cli --fallback gemini_3.8
```

CodeMaster non installa le CLI e non effettua il login. Il backend avvia la CLI selezionata in una cartella temporanea e passa le domande via stdin. Per le richieste che richiedono lettura della repo, espone via MCP solo gli strumenti attivi per modalità e domanda; `web_lookup` e `run_command` restano subordinati a `--allow-web` e `--allow-exec`. L'endpoint MCP ascolta solo su loopback e usa una porta e un bearer temporanei. Le variabili d'ambiente con suffisso `_API_KEY` o `_API_TOKEN` vengono rimosse dal processo CLI, che usa quindi la propria sessione autenticata.

Per Mistral Vibe completa il setup preventivamente con `vibe --setup`: usa l'accesso browser o salva la credenziale nel keyring/file `.vibe` dell'utente, perché le variabili API della shell non vengono inoltrate a CodeMaster.

`--fallback` accetta solo alias API: un errore della CLI usa API a consumo solo quando il fallback è configurato. Senza fallback viene mostrato l'errore della CLI. L'uso dei provider CLI richiede un ambiente locale con binario e login disponibili; Kaggle, Colab e `--share` Gradio non sono supportati. Le versioni delle CLI possono cambiare opzioni o configurazione MCP: aggiornale e verifica `--help` se CodeMaster segnala un errore di avvio.

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
