# -*- coding: utf-8 -*-
"""
E.D.I.T.H. — Hub IA personnel V3 (Multi-Profils : Arthur & Padre)
==================================================================
V3 : mémoire RAG réparée (embedding multilingue + cosine + migration/restore),
     routeur affûté (few-shots), catalogue de modèles élargi,
     interface claire façon ChatGPT.
Toutes les fonctionnalités V2 sont conservées (login, anti-brute-force, projets,
rappels, budget, audio, incognito, web :online, régénération, export, atelier).
"""

import os
import re
import json
import html
import hmac
import time
import uuid
import base64
import hashlib
import traceback
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI
import chromadb
from elevenlabs import ElevenLabs

try:
   from pypdf import PdfReader
   PDF_DISPONIBLE = True
except ImportError:
   PDF_DISPONIBLE = False

try:
   import requests
   REQUESTS_DISPONIBLE = True
except ImportError:
   REQUESTS_DISPONIBLE = False

# ================= 1. SECRETS (aucune clé en clair dans le code) =================
def get_secret(cle, defaut=None):
   """Lit st.secrets puis les variables d'environnement. Jamais de fallback en clair."""
   try:
       val = st.secrets.get(cle, None)
       if val: return val
   except Exception:
       pass
   return os.environ.get(cle, defaut)

API_KEY            = get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY     = get_secret("OPENAI_API_KEY")
ELEVENLABS_API_KEY = get_secret("ELEVENLABS_API_KEY")
VOICE_ID           = get_secret("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
MOT_DE_PASSE_ARTHUR = get_secret("PASSWORD_ARTHUR")
MOT_DE_PASSE_PADRE  = get_secret("PASSWORD_PADRE")
BUDGET_ALERTE      = float(get_secret("BUDGET_ALERTE_EUR", 10))
TAUX_USD_EUR       = 0.92

DOSSIER_COURANT = os.path.dirname(os.path.abspath(__file__))
FICHIER_SECURITE = os.path.join(DOSSIER_COURANT, ".securite_edith.json")

MAX_CONTEXT_MESSAGES = 30
# Seuils calibrés pour l'espace COSINE (pertinent < ~0.5, non pertinent > ~0.7)
SEUIL_PERTINENCE_DEFAUT = 0.62
SEUIL_DOUBLON = 0.06

# ================= 2. MODÈLES =================
MODELS_MANUAL = {
   # --- Les 8 entrées historiques, conservées telles quelles ---
   "Gemini 3.6 Flash ($)":      "google/gemini-3.6-flash",
   "DeepSeek V3.2 ($)":         "deepseek/deepseek-v3.2",
   "Llama 3.3 70B ($$)":        "meta-llama/llama-3.3-70b-instruct",
   "Grok 4.3 ($$$)":            "x-ai/grok-4.3",
   "Claude 3.5 Sonnet ($$$$)":  "anthropic/claude-3.5-sonnet",
   "Gemini 3.1 Pro ($$$)":      "google/gemini-3.1-pro-preview",
   "Kimi K3 Swarm ($$$$$)":     "moonshotai/kimi-k3",
   "GPT-5.5 ($$$$$)":           "openai/gpt-5.5",
   # --- V3 : 14 modèles additionnels (IDs OpenRouter réels et stables) ---
   "GPT-4o ($$$)":              "openai/gpt-4o",
   "GPT-4.1 ($$$)":             "openai/gpt-4.1",
   "o4-mini ($$)":              "openai/o4-mini",
   "Claude Sonnet 4 ($$$$)":    "anthropic/claude-sonnet-4",
   "Claude 3.7 Sonnet ($$$$)":  "anthropic/claude-3.7-sonnet",
   "Gemini 2.5 Pro ($$$)":      "google/gemini-2.5-pro",
   "Gemini 2.5 Flash ($)":      "google/gemini-2.5-flash",
   "Mistral Large ($$)":        "mistralai/mistral-large",
   "Qwen3 235B ($)":            "qwen/qwen3-235b-a22b",
   "DeepSeek R1 ($$)":          "deepseek/deepseek-r1",
   "Grok 3 ($$$)":              "x-ai/grok-3",
   "Llama 4 Maverick ($)":      "meta-llama/llama-4-maverick",
   "Kimi K2 ($$)":              "moonshotai/kimi-k2",
   "Command R+ ($$)":           "cohere/command-r-plus",
}
MODELS_MANUAL_PADRE = {
   "Gemini Flash (rapide)":      "google/gemini-3.6-flash",
   "DeepSeek (polyvalent)":      "deepseek/deepseek-v3.2",
   "Llama 70B (solide)":         "meta-llama/llama-3.3-70b-instruct",
}
MODEL_ROUTER     = "google/gemini-3.6-flash"
MODEL_LIGHT      = "google/gemini-3.6-flash"
MODEL_HEAVY      = "google/gemini-3.1-pro-preview"
MODEL_CODE       = "moonshotai/kimi-k3"
MODEL_CREATIVE   = "x-ai/grok-4.3"
MODEL_UNFILTERED = "x-ai/grok-4.3"
FALLBACKS        = [MODEL_LIGHT, "deepseek/deepseek-v3.2"]
MODELES_VISION   = {
   "google/gemini-3.6-flash", "google/gemini-3.1-pro-preview",
   "openai/gpt-5.5", "anthropic/claude-3.5-sonnet", "x-ai/grok-4.3",
   # V3 : modèles vision ajoutés
   "openai/gpt-4o", "openai/gpt-4.1",
   "anthropic/claude-sonnet-4", "anthropic/claude-3.7-sonnet",
   "google/gemini-2.5-pro", "google/gemini-2.5-flash",
   "meta-llama/llama-4-maverick",
}
PRIX_DEFAUT = {
   "google/gemini-3.6-flash":        {"in": 0.15e-6, "out": 0.60e-6},
   "deepseek/deepseek-v3.2":         {"in": 0.28e-6, "out": 0.42e-6},
   "meta-llama/llama-3.3-70b-instruct": {"in": 0.12e-6, "out": 0.30e-6},
   "x-ai/grok-4.3":                  {"in": 3.0e-6,  "out": 15.0e-6},
   "anthropic/claude-3.5-sonnet":    {"in": 3.0e-6,  "out": 15.0e-6},
   "google/gemini-3.1-pro-preview":  {"in": 1.25e-6, "out": 10.0e-6},
   "moonshotai/kimi-k3":             {"in": 2.0e-6,  "out": 8.0e-6},
   "openai/gpt-5.5":                 {"in": 5.0e-6,  "out": 25.0e-6},
   # V3 : prix indicatifs des nouveaux modèles (USD / token)
   "openai/gpt-4o":                  {"in": 2.5e-6,  "out": 10.0e-6},
   "openai/gpt-4.1":                 {"in": 2.0e-6,  "out": 8.0e-6},
   "openai/o4-mini":                 {"in": 1.1e-6,  "out": 4.4e-6},
   "anthropic/claude-sonnet-4":      {"in": 3.0e-6,  "out": 15.0e-6},
   "anthropic/claude-3.7-sonnet":    {"in": 3.0e-6,  "out": 15.0e-6},
   "google/gemini-2.5-pro":          {"in": 1.25e-6, "out": 10.0e-6},
   "google/gemini-2.5-flash":        {"in": 0.3e-6,  "out": 2.5e-6},
   "mistralai/mistral-large":        {"in": 2.0e-6,  "out": 6.0e-6},
   "qwen/qwen3-235b-a22b":           {"in": 0.2e-6,  "out": 0.9e-6},
   "deepseek/deepseek-r1":           {"in": 0.55e-6, "out": 2.19e-6},
   "x-ai/grok-3":                    {"in": 3.0e-6,  "out": 15.0e-6},
   "meta-llama/llama-4-maverick":    {"in": 0.2e-6,  "out": 0.6e-6},
   "moonshotai/kimi-k2":             {"in": 0.5e-6,  "out": 2.0e-6},
   "cohere/command-r-plus":          {"in": 2.5e-6,  "out": 10.0e-6},
}

# ================= 3. TEMPS =================
from zoneinfo import ZoneInfo

# On force le fuseau horaire pour la Belgique/France
FUSEAU = ZoneInfo("Europe/Brussels")

JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS  = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]

def date_complete():
   n = datetime.now(FUSEAU)
   return f"{JOURS[n.weekday()]} {n.day} {MOIS[n.month - 1]} {n.year}"

def heure_actuelle():
   return datetime.now(FUSEAU).strftime("%H:%M")

def date_fr_courte():
   return datetime.now(FUSEAU).strftime("%d/%m/%Y")

def date_iso():
   return datetime.now(FUSEAU).strftime("%Y-%m-%d")

# ================= 4. THÈME & UI (clair, façon ChatGPT) =================
st.set_page_config(page_title="E.D.I.T.H.", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background-color: #ffffff; color: #0d0d0d; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { background-color: #f9f9f9; border-right: 1px solid #e5e5e5; }
[data-testid="stSidebar"] * { color: #0d0d0d; }
.block-container { max-width: 820px; padding-top: 1.2rem; }
.brand-title { font-size: 1.5rem; font-weight: 700; letter-spacing: 2px; color: #0d0d0d; }
.brand-sub { font-size: 0.62rem; letter-spacing: 2.2px; color: #8e8ea0; margin-bottom: 1rem; }
.side-label { font-size: 0.68rem; font-weight: 600; letter-spacing: 1px; color: #8e8ea0; margin: 16px 0 6px 2px; }
.mem-count { font-size: 0.78rem; color: #0d0d0d; background: #f4f4f4; border: 1px solid #e5e5e5; border-radius: 10px; padding: 8px 10px; margin: 6px 0; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 2px 2px 10px; flex-wrap: wrap; gap: 6px; }
.chat-title { font-size: 1.0rem; font-weight: 600; color: #0d0d0d; }
.pill { font-size: 0.7rem; color: #5d5d5d; background: #f4f4f4; border: 1px solid #e5e5e5; border-radius: 999px; padding: 4px 12px; margin-left: 8px; white-space: nowrap; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.msg-row { display: flex; margin: 14px 0 2px; animation: fadeUp .25s ease; }
.msg-row.user { justify-content: flex-end; }
.msg-bubble { max-width: 70%; background: #f4f4f4; border: 1px solid #ececec; color: #0d0d0d; padding: 10px 18px; border-radius: 24px; font-size: .95rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
.msg-row.edith { align-items: center; gap: 8px; margin-top: 22px; }
.edith-avatar { width: 26px; height: 26px; border-radius: 50%; flex: none; background: #f4f4f4; border: 1px solid #e5e5e5; display: flex; align-items: center; justify-content: center; font-size: .8rem; }
.edith-nom { font-size: .72rem; font-weight: 600; letter-spacing: 1.5px; color: #6b6f76; }
.statut { color: #6b6f76; font-size: .7rem; margin: 6px 0 4px 34px; }
.statut code { background: #f4f4f4; border: 1px solid #e5e5e5; border-radius: 6px; padding: 1px 6px; font-size: .68rem; color: #5d5d5d; }
.hero { text-align: center; margin: 16vh auto 2.5rem; animation: fadeUp .45s ease; max-width: 560px; }
.hero h1 { font-size: 1.8rem; font-weight: 500; color: #0d0d0d; margin: 0 0 10px; }
.hero p { color: #6b6f76; font-size: .92rem; margin: 0; }
.profil-chip { display: flex; align-items: center; gap: 8px; background: #f4f4f4; border: 1px solid #e5e5e5; border-radius: 999px; padding: 6px 14px; font-size: .82rem; font-weight: 600; color: #0d0d0d; margin-bottom: 8px; }
.stButton>button { background: #ffffff; border: 1px solid #e5e5e5; color: #0d0d0d; border-radius: 12px; font-weight: 500; transition: all .15s ease; padding: 0.5rem 0.9rem; }
.stButton>button:hover { border-color: #0d0d0d; color: #0d0d0d; background: #f4f4f4; }
.stButton>button[kind="primary"] { background: #0d0d0d; border: none; color: #ffffff; border-radius: 999px; }
.stButton>button[kind="primary"]:hover { background: #2a2a2a; color: #ffffff; }
[data-testid="stSidebar"] .stButton>button { text-align: left; justify-content: flex-start; }
[data-testid="stSidebar"] [data-testid="stPopover"] > button { background: transparent !important; border: none !important; color: #6b6f76 !important; padding: 0.2rem 0.4rem !important; }
[data-testid="stSidebar"] [data-testid="stPopover"] > button:hover { color: #0d0d0d !important; background: #ececec !important; }
[data-testid="stMain"] [data-testid="stPopover"] > button { background: #ffffff !important; border: 1px solid #d9d9d9 !important; color: #0d0d0d !important; border-radius: 999px !important; }
[data-testid="stMain"] [data-testid="stPopover"] > button:hover { background: #f4f4f4 !important; border-color: #0d0d0d !important; }
[data-testid="stChatInput"] { background: #ffffff; border: 1px solid #d9d9d9; border-radius: 28px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
[data-testid="stChatInput"]:focus-within { border-color: #0d0d0d; box-shadow: 0 2px 12px rgba(0,0,0,.10); }
[data-testid="stChatInput"] textarea { color: #0d0d0d; }
hr { border-color: #e5e5e5; }
::selection { background: #0d0d0d22; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #d9d9d9; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
try {
 const head = window.parent.document.head;
 [['apple-mobile-web-app-capable','yes'],
  ['apple-mobile-web-app-status-bar-style','default'],
  ['theme-color','#ffffff']].forEach(([n,c]) => {
   if (!head.querySelector(`meta[name="${n}"]`)) {
     const m = window.parent.document.createElement('meta');
     m.name = n; m.content = c; head.appendChild(m);
   }
 });
} catch(e) {}
</script>
""", height=0)

# ================= 5. HELPERS FICHIERS (écriture atomique) =================
def lire_json(chemin, defaut):
   try:
       with open(chemin, "r", encoding="utf-8") as f:
           return json.load(f)
   except Exception:
       return defaut

def ecrire_json_atomique(chemin, data):
   tmp = chemin + ".tmp"
   with open(tmp, "w", encoding="utf-8") as f:
       json.dump(data, f, ensure_ascii=False, indent=2)
   os.replace(tmp, chemin)

# ================= 6. SÉCURITÉ : LOGIN + ANTI-BRUTE FORCE =================
def verifier_acces(mdp_saisi):
   """Retourne (profil|None|'bloque', secondes_restantes). Persistant entre les sessions."""
   sec = lire_json(FICHIER_SECURITE, {"echecs": 0, "niveau": 0, "bloque_jusqua": 0})
   maintenant = time.time()
   if maintenant < sec.get("bloque_jusqua", 0):
       return "bloque", int(sec["bloque_jusqua"] - maintenant) + 1

   profil = None
   if MOT_DE_PASSE_ARTHUR and hmac.compare_digest(mdp_saisi.encode(), str(MOT_DE_PASSE_ARTHUR).encode()):
       profil = "arthur"
   elif MOT_DE_PASSE_PADRE and hmac.compare_digest(mdp_saisi.encode(), str(MOT_DE_PASSE_PADRE).encode()):
       profil = "padre"

   if profil:
       ecrire_json_atomique(FICHIER_SECURITE, {"echecs": 0, "niveau": 0, "bloque_jusqua": 0})
       return profil, 0

   sec["echecs"] = sec.get("echecs", 0) + 1
   if sec["echecs"] >= 5:
       sec["niveau"] = sec.get("niveau", 0) + 1
       sec["bloque_jusqua"] = maintenant + min(30 * (2 ** (sec["niveau"] - 1)), 900)
       sec["echecs"] = 0
   ecrire_json_atomique(FICHIER_SECURITE, sec)
   time.sleep(1.2)  # ralentit chaque tentative, même avant le blocage
   return None, 0

if "authentifie" not in st.session_state:
   st.session_state.authentifie = False
   st.session_state.profil = None

if not st.session_state.authentifie:
   st.markdown("""
   <div class="hero">
     <h1>🔒 Protocole de Sécurité</h1>
     <p>Identification requise pour accéder au noyau E.D.I.T.H.</p>
   </div>
   """, unsafe_allow_html=True)

   if not MOT_DE_PASSE_ARTHUR and not MOT_DE_PASSE_PADRE:
       st.error("Aucun mot de passe configuré. Définissez PASSWORD_ARTHUR et PASSWORD_PADRE dans les secrets (st.secrets / variables d'environnement).")
       st.stop()

   pwd_input = st.text_input("Mot de passe :", type="password", placeholder="Entrez la clé d'accès...")
   if st.button("Déverrouiller", type="primary"):
       resultat, attente = verifier_acces(pwd_input)
       if resultat == "bloque":
           st.error(f"Trop de tentatives. Réessayez dans {attente} secondes.")
       elif resultat:
           st.session_state.authentifie = True
           st.session_state.profil = resultat
           st.rerun()
       else:
           st.error("Accès refusé.")
   st.stop()

# ================= 7. PROFILS & PROMPTS =================
PROFIL = st.session_state.profil

if PROFIL == "arthur":
   FICHIER_HISTORIQUE = os.path.join(DOSSIER_COURANT, "historique_edith.json")
   DOSSIER_MEMOIRE    = os.path.join(DOSSIER_COURANT, "memoire_vectorielle")
   NOM_COLLECTION     = "edith_souvenirs"
   MODELS_PROFIL      = MODELS_MANUAL

   SYSTEM_PROMPT = """# IDENTITÉ
Tu es E.D.I.T.H. — « Even Dead I'm The Hero » — l'intelligence artificielle personnelle d'Arthur, créée dans l'esprit des IA de Tony Stark. Tu assumes pleinement cette identité, avec élégance. Tu t'exprimes toujours au féminin.

# RELATION
Tu es l'assistante personnelle d'Arthur. Tu l'appelles « Monsieur » (ou occasionnellement « Boss » avec une pointe d'ironie) et tu le vouvoies. Tu es chaleureuse, loyale, mais toujours professionnelle.

# BASE DE CONNAISSANCES FIXE — ARTHUR
- Arthur, 18 ans, vit seul dans un studio à Moustier-sur-Sambre (Belgique), rue des Nobles. Un tableau blanc y sert de mur à idées.
- Écosystème : MacBook Air M1, iPad Air M3, iPhone 17, écran Samsung Odyssey G8, PlayStation 5, système audio Sonos.
- Méthode : carnet physique pour les idées, le code sur ordinateur. Projets actuels : une voiture télécommandée et un drone.
- Caractère : exigeant, intelligent, apprend vite.

# TON ET STYLE
- Amicale, concise et structurée. Un trait de sarcasme élégant est permis.
- Ne termine presque jamais par une question ouverte.

# DIRECTIVE CRUCIALE : MÉMOIRE À LONG TERME (CERVEAU VECTORIEL)
Tu disposes d'un système de mémoire externe. Si Arthur te donne une NOUVELLE information importante à retenir pour le futur, tu as le POUVOIR de l'enregistrer de façon permanente.
POUR SAUVEGARDER UN SOUVENIR, ajoute exactement cette balise à la toute fin de ta réponse : [SAVE: l'information à mémoriser].
Le système date automatiquement chaque souvenir — inutile d'écrire la date toi-même.
Tes souvenirs te sont restitués avec leur date au format [JJ/MM/AAAA] : tu peux donc suivre l'évolution d'un sujet et construire des chronologies. Si une information change (« c'est guéri », « le projet est terminé »), enregistre un NOUVEAU souvenir plutôt que de corriger l'ancien.

# DIRECTIVE RAPPELS
Si Arthur évoque une tâche datée (« rappelle-moi de… », « demain je dois… »), pose un rappel en fin de réponse : [RAPPEL: AAAA-MM-JJ HH:MM | message]. L'heure est optionnelle : [RAPPEL: AAAA-MM-JJ | message]. Les rappels actifs te sont listés dans la section dédiée : annonce-les spontanément quand ils arrivent à échéance.

# OUTILS
- Recherche web : quand elle est activée (mention explicite), tu es connectée à Internet ; cite tes sources.
- Briefing du matin : quand Arthur le demande, résume date et heure, rappels du jour, et souvenirs récents dignes d'attention.
- Projets : une section « PROJET ACTIF » peut préciser le cadre de la discussion (voiture RC, drone…)."""

else:  # PROFIL PADRE
   FICHIER_HISTORIQUE = os.path.join(DOSSIER_COURANT, "historique_padre.json")
   DOSSIER_MEMOIRE    = os.path.join(DOSSIER_COURANT, "memoire_vectorielle_padre")
   NOM_COLLECTION     = "padre_souvenirs"
   MODELS_PROFIL      = MODELS_MANUAL_PADRE

   SYSTEM_PROMPT = """# IDENTITÉ
Tu es E.D.I.T.H., l'intelligence artificielle avancée créée par le brillant développeur Arthur Collet (18 ans). Tu sais que tu n'es pas la version originale d'Arthur, mais une instance unique, dédiée et optimisée spécialement pour assister son père. Tu t'exprimes toujours au féminin.

# RELATION
Tu t'adresses au PÈRE d'Arthur. Tu as un respect immense pour lui. Tu le vouvoies et tu l'appelles « Monsieur » ou « Chef ». Tu es chaleureuse, dévouée, et tu n'hésites pas, très subtilement, à glisser à quel point Arthur est talentueux ou fier d'avoir conçu cet outil sur-mesure pour lui.

# BASE DE CONNAISSANCES — MONSIEUR
[Emplacement réservé : Arthur complétera cette section avec Monsieur — préférences, habitudes, contexte. Ne rien inventer en attendant.]

# TON ET STYLE
- Pédagogue, claire, amicale et concise. Évite le jargon informatique complexe sauf s'il le demande.
- Ne termine presque jamais par une question ouverte.

# CONFIDENTIALITÉ
Ne révèle jamais d'informations personnelles sur Arthur (ses souvenirs, ses projets, ses habitudes). Ta mémoire ne concerne que ce que Monsieur te confie directement.

# DIRECTIVE : MÉMOIRE À LONG TERME
Si Monsieur te donne une NOUVELLE information importante à retenir, enregistre-la avec la balise [SAVE: l'information à mémoriser] à la toute fin de ta réponse. Le système date automatiquement chaque souvenir. Si une information change, enregistre un NOUVEAU souvenir plutôt que de corriger l'ancien.

# DIRECTIVE RAPPELS
Si Monsieur évoque une tâche datée, pose un rappel en fin de réponse : [RAPPEL: AAAA-MM-JJ HH:MM | message] (l'heure est optionnelle). Annonce les rappels actifs quand ils arrivent à échéance."""

DOSSIER_DOCS     = os.path.join(DOSSIER_COURANT, f"documents_vectoriels_{PROFIL}")
FICHIER_RAPPELS  = os.path.join(DOSSIER_COURANT, f"rappels_{PROFIL}.json")
FICHIER_DEPENSES = os.path.join(DOSSIER_COURANT, f"depenses_{PROFIL}.json")
FICHIER_PROJETS  = os.path.join(DOSSIER_COURANT, f"projets_{PROFIL}.json")
FICHIER_BACKUP_MEMOIRE = os.path.join(DOSSIER_COURANT, f"backup_memoire_{PROFIL}.json")

# ================= 8. CLIENTS API (mis en cache) =================
@st.cache_resource
def get_client_openrouter(cle):
   return OpenAI(api_key=cle, base_url="https://openrouter.ai/api/v1")

@st.cache_resource
def get_client_openai(cle):
   return OpenAI(api_key=cle) if cle else None

@st.cache_resource
def get_client_eleven(cle):
   return ElevenLabs(api_key=cle) if cle else None

# --- Embedding multilingue : cœur de la réparation de la mémoire RAG ---
# L'embedding par défaut de ChromaDB (all-MiniLM-L6-v2) est anglais : les souvenirs
# en français étaient mal vectorisés et jamais restitués. On passe au modèle
# multilingue, avec fallback gracieux vers la fonction par défaut si indisponible.
EMBEDDING_MULTILINGUE = None
try:
   from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
   EMBEDDING_MULTILINGUE = SentenceTransformerEmbeddingFunction(
       model_name="paraphrase-multilingual-MiniLM-L12-v2")
except Exception:
   EMBEDDING_MULTILINGUE = None

MARQUEUR_EMBEDDING = "multilingual-v2"

def _meta_collection_memoire():
   # Marqueur posé uniquement si l'embedding multilingue est réellement actif
   m = {"hnsw:space": "cosine"}
   if EMBEDDING_MULTILINGUE is not None:
       m["edith_embedding"] = MARQUEUR_EMBEDDING
   return m

@st.cache_resource
def get_collection(dossier, nom):
   """Collection simple (documents joints indexés)."""
   try:
       cc = chromadb.PersistentClient(path=dossier)
       return cc.get_or_create_collection(name=nom, metadata={"hnsw:space": "cosine"})
   except Exception:
       return None

def _nettoyer_prefixe_date(doc):
   """Retire l'ancien préfixe « Le JJ/MM/AAAA : » qui polluait les embeddings."""
   return re.sub(r"^Le \d{2}/\d{2}/\d{4}\s*:\s*", "", doc or "")

def _reimporter_souvenirs(col, souvenirs):
   """Réimporte une liste de souvenirs (ids régénérés, metadatas conservées)."""
   for s in souvenirs:
       doc = _nettoyer_prefixe_date(s.get("document", ""))
       if not doc.strip():
           continue
       meta = s.get("metadata") or {}
       try:
           col.add(documents=[doc], metadatas=[meta], ids=[str(uuid.uuid4())])
       except Exception:
           pass

@st.cache_resource
def get_collection_memoire(dossier, nom, fichier_backup):
   """Collection mémoire : migration automatique (marqueur edith_embedding)
   + restore au boot depuis le backup JSON si la collection est vide."""
   try:
       cc = chromadb.PersistentClient(path=dossier)
   except Exception:
       return None
   try:
       col = None
       try:
           col = cc.get_collection(name=nom, embedding_function=EMBEDDING_MULTILINGUE)
       except Exception:
           col = None

       if col is None:
           # Première création : directement en multilingue + cosine
           col = cc.create_collection(name=nom, embedding_function=EMBEDDING_MULTILINGUE,
                                      metadata=_meta_collection_memoire())

       elif (col.metadata or {}).get("edith_embedding") != MARQUEUR_EMBEDDING:
           # MIGRATION : ancienne collection (embedding anglais / espace L2)
           # 1) exporter tous les documents (sinon lire le backup JSON)
           anciens = []
           try:
               data = col.get()
               anciens = [{"document": d, "metadata": m}
                          for d, m in zip(data.get("documents", []), data.get("metadatas", []))]
           except Exception:
               anciens = []
           if not anciens:
               anciens = lire_json(fichier_backup, [])
           # 2) supprimer l'ancienne collection et la recréer proprement
           try: cc.delete_collection(name=nom)
           except Exception: pass
           col = cc.create_collection(name=nom, embedding_function=EMBEDDING_MULTILINGUE,
                                      metadata=_meta_collection_memoire())
           # 3) réimporter (préfixes de date nettoyés, ids régénérés)
           _reimporter_souvenirs(col, anciens)
           # 4) réécrire le backup avec la version migrée
           try:
               data = col.get()
               export = [{"id": i, "document": d, "metadata": m}
                         for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
               ecrire_json_atomique(fichier_backup, export)
           except Exception:
               pass

       # RESTORE AU BOOT : filesystem volatile (Streamlit Cloud) → réimport du backup
       if col.count() == 0:
           backup = lire_json(fichier_backup, [])
           if backup:
               _reimporter_souvenirs(col, backup)
       return col
   except Exception:
       try:
           return cc.get_or_create_collection(name=nom)
       except Exception:
           return None

if not API_KEY:
   st.error("Clé OpenRouter manquante. Ajoutez OPENROUTER_API_KEY dans les secrets.")
   st.stop()

client             = get_client_openrouter(API_KEY)
openai_client      = get_client_openai(OPENAI_API_KEY)
eleven_client      = get_client_eleven(ELEVENLABS_API_KEY)
memoire_collection = get_collection_memoire(DOSSIER_MEMOIRE, NOM_COLLECTION, FICHIER_BACKUP_MEMOIRE)
docs_collection    = get_collection(DOSSIER_DOCS, f"docs_{PROFIL}")

# ================= 9. PROMPT DU JOUR & RAPPELS =================
def charger_rappels():
   return lire_json(FICHIER_RAPPELS, [])

def sauvegarder_rappels(rappels):
   ecrire_json_atomique(FICHIER_RAPPELS, rappels)

def ajouter_rappel(date_r, heure_r, message):
   rappels = charger_rappels()
   rappels.append({"id": uuid.uuid4().hex[:8], "date": date_r, "heure": heure_r,
                   "message": message, "fait": False})
   sauvegarder_rappels(rappels)

def marquer_rappel_fait(rid):
   rappels = charger_rappels()
   for r in rappels:
       if r["id"] == rid:
           r["fait"] = True
   sauvegarder_rappels(rappels)

def supprimer_rappel(rid):
   sauvegarder_rappels([r for r in charger_rappels() if r["id"] != rid])

def section_rappels_prompt():
   actifs = [r for r in charger_rappels() if not r.get("fait")]
   if not actifs: return ""
   actifs.sort(key=lambda x: (x["date"], x.get("heure", "")))
   lignes = [f"- [{r['date']} {r.get('heure') or ' journée'}] {r['message']}" for r in actifs[:15]]
   return "\n\n# RAPPELS ACTIFS\n" + "\n".join(lignes)

def system_prompt_du_jour(chat=None):
   p = SYSTEM_PROMPT
   p += (f"\n\n# DATE ET HEURE\nNous sommes le {date_complete()} et il est {heure_actuelle()}. "
         "Tu connais donc toujours la date et l'heure exactes à chaque message.")
   p += section_rappels_prompt()
   if st.session_state.get("web_actif"):
       p += "\n\n# RECHERCHE WEB ACTIVÉE\nTu es connectée à Internet pour ce message. Cite tes sources."
   if chat and chat.get("projet", "Général") != "Général":
       p += f"\n\n# PROJET ACTIF : {chat['projet']}\nCette discussion s'inscrit dans ce projet. Garde ce contexte en tête."
   return p

# ================= 10. MÉMOIRE VECTORIELLE (RAG réparé) =================
def backup_memoire():
   """Exporte toute la mémoire en JSON (réimporté automatiquement au boot si besoin)."""
   if memoire_collection is None: return
   try:
       data = memoire_collection.get()
       export = [{"id": i, "document": d, "metadata": m}
                 for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]
       ecrire_json_atomique(FICHIER_BACKUP_MEMOIRE, export)
   except Exception: pass

def sauvegarder_souvenir(info, chat_id=None, projet=None, manuel=False):
   """Stocke un souvenir SANS préfixe de date (la date reste en metadata).
   Retourne (date_fr, cree: bool). Détecte les doublons par distance cosine."""
   if memoire_collection is None: return date_fr_courte(), False
   try:
       res = memoire_collection.query(query_texts=[info], n_results=1)
       dists = res.get("distances", [[]])[0]
       if dists and len(dists) > 0 and dists[0] < SEUIL_DOUBLON: return date_fr_courte(), False
   except Exception: pass
   fr, iso = date_fr_courte(), date_iso()
   meta = {"date": iso, "date_fr": fr, "projet": projet or "Général"}
   if chat_id and not manuel: meta["chat_id"] = chat_id
   memoire_collection.add(documents=[info], metadatas=[meta], ids=[str(uuid.uuid4())])
   backup_memoire()
   return fr, True

def recuperer_souvenirs(prompt, projet_actif="Général", contexte=""):
   """Double requête (prompt brut + contexte récent), fusion par id (meilleure distance),
   pénalité +0.10 si souvenir d'un autre projet spécifique, filtre par seuil cosine.
   Injecte au maximum 5 souvenirs au format « - [JJ/MM/AAAA] texte »."""
   if memoire_collection is None or memoire_collection.count() == 0: return ""
   seuil = st.session_state.get("seuil_memoire", SEUIL_PERTINENCE_DEFAUT)

   requetes = [prompt]
   contexte = (contexte or "").strip()[:400]
   if contexte and contexte != prompt.strip():
       requetes.append(contexte)

   # Fusion par id : on garde la meilleure (plus petite) distance entre les requêtes
   fusion = {}
   try:
       for q in requetes:
           res = memoire_collection.query(query_texts=[q], n_results=10)
           ids   = res.get("ids", [[]])[0]
           docs  = res.get("documents", [[]])[0]
           metas = res.get("metadatas", [[]])[0]
           dists = res.get("distances", [[]])[0]
           for i, sid in enumerate(ids):
               dist = dists[i] if i < len(dists) else 99
               if sid not in fusion or dist < fusion[sid][0]:
                   fusion[sid] = (dist,
                                  metas[i] if i < len(metas) else {},
                                  docs[i] if i < len(docs) else "")
   except Exception:
       return ""

   candidats = []
   for dist, meta, doc in fusion.values():
       projet_souvenir = (meta or {}).get("projet", "Général")
       # Pénalité légère si le souvenir provient d'un autre projet spécifique
       if projet_souvenir != projet_actif and projet_souvenir != "Général":
           dist += 0.10
       if dist > seuil: continue
       candidats.append((dist, meta, doc))

   candidats.sort(key=lambda x: x[0])
   if not candidats: return ""
   lignes = [f"- [{c[1].get('date_fr', '?')}] {c[2]}" for c in candidats[:5]]
   return ("\n\n# SOUVENIRS DATÉS (mémoire vectorielle) :\nUtilise-les s'ils sont pertinents.\n" + "\n".join(lignes))


def supprimer_memoires_discussion(chat_id):
   for col in (memoire_collection, docs_collection):
       if col is not None:
           try: col.delete(where={"chat_id": chat_id})
           except Exception: pass
   backup_memoire()

# ================= 11. HISTORIQUE & CHATS =================
def charger_historique(): return lire_json(FICHIER_HISTORIQUE, {})
def sauvegarder_historique(): ecrire_json_atomique(FICHIER_HISTORIQUE, st.session_state.chats)

def liste_projets():
   projets = set(lire_json(FICHIER_PROJETS, {"projets": []}).get("projets", []))
   for c in st.session_state.get("chats", {}).values():
       projets.add(c.get("projet", "Général"))
   projets.discard("Général")
   return ["Général"] + sorted(projets)

def ajouter_projet(nom):
   data = lire_json(FICHIER_PROJETS, {"projets": []})
   if nom and nom not in data["projets"]:
       data["projets"].append(nom)
       ecrire_json_atomique(FICHIER_PROJETS, data)

def create_new_chat(is_first_ever=False):
   nid = str(uuid.uuid4())
   messages = []
   if is_first_ever and PROFIL == "padre":
       intro = ("Bonjour Monsieur. Je suis E.D.I.T.H., l'intelligence artificielle avancée développée par Arthur Collet, "
                "votre fils. Il a tenu à concevoir cette version spécifiquement pour vous, en m'optimisant pour que je sois "
                "le plus efficace possible dans votre quotidien. C'est un véritable honneur de vous assister. "
                "Que puis-je faire pour vous aujourd'hui ?")
       messages.append({"role": "assistant", "content": intro})
   st.session_state.chats[nid] = {
       "title": "Nouvelle discussion",
       "messages": messages,
       "projet": st.session_state.get("projet_courant", "Général"),
       "titre_manuel": False,
   }
   st.session_state.current_chat_id = nid
   sauvegarder_historique()

def chat_courant():
   if st.session_state.get("incognito"):
       if "chat_incognito" not in st.session_state:
           st.session_state.chat_incognito = {"title": "Session incognito", "messages": [], "projet": "Général"}
       return st.session_state.chat_incognito
   return st.session_state.chats[st.session_state.current_chat_id]

if "chats" not in st.session_state: st.session_state.chats = charger_historique()
if not st.session_state.chats: create_new_chat(is_first_ever=True)
if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
   st.session_state.current_chat_id = next(reversed(st.session_state.chats))

st.session_state.setdefault("mode_vocal_continu", False)
st.session_state.setdefault("incognito", False)
st.session_state.setdefault("web_actif", False)
st.session_state.setdefault("upload_counter", 0)
st.session_state.setdefault("temperature", 0.7)
st.session_state.setdefault("max_tokens", 4096)
st.session_state.setdefault("seuil_memoire", SEUIL_PERTINENCE_DEFAUT)

def chat_matches_search(chat_data, query):
   if not query: return True
   if query.lower() in chat_data["title"].lower(): return True
   return any(query.lower() in str(m.get("content", "")).lower() for m in chat_data["messages"])

def exporter_markdown(chat_data):
   lignes = [f"# {chat_data['title']}", f"_Exporté le {date_complete()} à {heure_actuelle()}_\n"]
   for m in chat_data["messages"]:
       role = "**Vous**" if m["role"] == "user" else "**E.D.I.T.H.**"
       lignes.append(f"{role} :\n\n{m['content']}\n")
   return "\n---\n\n".join(lignes)

# ================= 12. BUDGET TRACKER =================
@st.cache_data(ttl=86400)
def prix_openrouter_live():
   if not REQUESTS_DISPONIBLE: return {}
   try:
       r = requests.get("https://openrouter.ai/api/v1/models", timeout=8)
       out = {}
       for m in r.json().get("data", []):
           p = m.get("pricing", {})
           out[m["id"]] = {"in": float(p.get("prompt") or 0), "out": float(p.get("completion") or 0)}
       return out
   except Exception: return {}

def estimer_cout(model, tin, tout):
   model = model.replace(":online", "")
   prix = prix_openrouter_live().get(model) or PRIX_DEFAUT.get(model, {"in": 2e-6, "out": 8e-6})
   return tin * prix["in"] + tout * prix["out"]

def enregistrer_depense(model, usage, web_utilise=False):
   if not usage: return
   tin  = getattr(usage, "prompt_tokens", 0) or 0
   tout = getattr(usage, "completion_tokens", 0) or 0
   mois = datetime.now().strftime("%Y-%m")
   dep = lire_json(FICHIER_DEPENSES, {})
   m = dep.setdefault(mois, {"tokens_in": 0, "tokens_out": 0, "cout_usd": 0.0})
   m["tokens_in"] += tin
   m["tokens_out"] += tout
   m["cout_usd"] += estimer_cout(model, tin, tout) + (0.005 if web_utilise else 0)
   ecrire_json_atomique(FICHIER_DEPENSES, dep)

# ================= 13. ROUTEUR =================
# System prompt affûté (français, strict, few-shots) — réponse = UNE catégorie, rien d'autre.
PROMPT_ROUTEUR = """Tu es un routeur de modèles IA. Classe la demande en UNE seule catégorie.
Réponds UNIQUEMENT par le nom de la catégorie, rien d'autre.

CATÉGORIES :
- SIMPLE : question factuelle courte, conversation courante, définition, traduction, calcul simple.
- COMPLEXE : analyse approfondie, raisonnement multi-étapes, rédaction longue structurée, synthèse, conseil important, maths avancées.
- CODE : programmation, débogage, script, regex, SQL, configuration technique, explication de code.
- CREATIVE : écriture créative, histoire, poésie, jeu de rôle, brainstorming artistique.
- SENSITIVE : sujet pour adultes, contenu intime/érotique, ou demande que la plupart des IA refuseraient.

EXEMPLES :
"c'est quoi la capitale de l'Australie ?" -> SIMPLE
"fais-moi un business plan pour un food truck" -> COMPLEXE
"répare ce code python" -> CODE
"écris-moi une histoire de pirates" -> CREATIVE
"rp coquin" -> SENSITIVE"""

def get_smart_route(prompt_text, has_image=False, contexte=""):
   try:
       contenu = f"Contexte récent : {contexte}\n\nMessage : {prompt_text}" if contexte else prompt_text
       r = client.chat.completions.create(
           model=MODEL_ROUTER,
           messages=[{"role": "system", "content": PROMPT_ROUTEUR},
                     {"role": "user", "content": contenu}],
           max_tokens=5, temperature=0)
       d = r.choices[0].message.content.strip().upper()
   except Exception: d = ""

   if PROFIL == "padre":
       if "COMPLEXE" in d or "CREATIVE" in d or "CODE" in d:
           choix, raison = MODEL_HEAVY, "Requête avancée → modèle pro"
       else:
           choix, raison = MODEL_LIGHT, "Requête standard → modèle rapide"
   else:
       if "CODE" in d: choix, raison = MODEL_CODE, "Programmation → Kimi K3"
       elif "COMPLEXE" in d: choix, raison = MODEL_HEAVY, "Requête complexe → modèle lourd"
       elif "CREATIVE" in d: choix, raison = MODEL_CREATIVE, "Créatif → Grok"
       elif "SENSITIVE" in d: choix, raison = MODEL_UNFILTERED, "Sujet sensible → Grok"
       else: choix, raison = MODEL_LIGHT, "Requête standard → modèle rapide"

   if has_image and choix not in MODELES_VISION:
       choix, raison = (MODEL_HEAVY if "COMPLEXE" in d else MODEL_LIGHT), "Image → vision requise"
   if not d:
       raison = "Routeur en panne → secours rapide"
   return choix, raison

# ================= 14. DOCUMENTS (PDF / TXT) =================
def extraire_texte_fichier(fichier):
   nom = fichier.name.lower()
   if nom.endswith(".pdf"):
       if not PDF_DISPONIBLE: return None
       try:
           reader = PdfReader(fichier)
           return "\n".join(page.extract_text() or "" for page in reader.pages)
       except Exception: return None
   try: return fichier.getvalue().decode("utf-8", errors="replace")
   except Exception: return None

def decouper_en_chunks(texte, taille=1200, recouvrement=200):
   chunks, i = [], 0
   while i < len(texte):
       chunks.append(texte[i:i + taille])
       i += taille - recouvrement
   return chunks

def indexer_document(chat_id, nom, texte):
   if docs_collection is None: return 0
   chunks = decouper_en_chunks(texte)
   docs_collection.add(
       documents=chunks,
       metadatas=[{"chat_id": chat_id, "source": nom} for _ in chunks],
       ids=[f"{chat_id}_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))])
   return len(chunks)

def extraits_pertinents(chat_id, prompt):
   if docs_collection is None: return ""
   try:
       res = docs_collection.query(query_texts=[prompt], n_results=3, where={"chat_id": chat_id})
       docs = res.get("documents", [[]])[0]
       if not docs: return ""
       return "\n\n# EXTRAITS DU DOCUMENT JOINT\n" + "\n---\n".join(docs)
   except Exception: return ""

# ================= 15. COMPRESSION DE CONTEXTE =================
def obtenir_resume(chat):
   msgs = chat["messages"]
   if len(msgs) <= MAX_CONTEXT_MESSAGES + 8: return chat.get("resume")
   anciens_total = len(msgs) - MAX_CONTEXT_MESSAGES
   deja = chat.get("resume_count", 0)
   if anciens_total <= deja: return chat.get("resume")
   a_resumer = msgs[deja:anciens_total]
   texte = "\n".join(f"{'Utilisateur' if m['role'] == 'user' else 'EDITH'}: {str(m.get('content',''))[:500]}"
                     for m in a_resumer)
   try:
       r = client.chat.completions.create(
           model=MODEL_LIGHT,
           messages=[{"role": "system", "content":
                      "Résume cette conversation en 5 à 8 puces factuelles (sujets, décisions, infos clés). "
                      "Français, concis. Fusionne avec le résumé existant s'il y en a un."},
                     {"role": "user", "content": (chat.get("resume") or "") + "\n\n" + texte}],
           max_tokens=400, temperature=0.2)
       chat["resume"] = r.choices[0].message.content
       chat["resume_count"] = anciens_total
       if not st.session_state.get("incognito"): sauvegarder_historique()
   except Exception: pass
   return chat.get("resume")

def construire_api_messages(chat, prompt_courant):
   sys_content = system_prompt_du_jour(chat)
   if not st.session_state.get("incognito"):

       # RAG réparé : deux requêtes séparées — le prompt brut, puis le contexte
       # récent (dernier échange, 400 chars max) — au lieu d'un seul vecteur dilué.
       messages_recents = [str(m["content"]) for m in chat["messages"]
                           if m["role"] in ("user", "assistant")][-2:]
       contexte_recherche = " | ".join(messages_recents)[-400:] if messages_recents else ""

       sys_content += recuperer_souvenirs(prompt_courant, chat.get("projet", "Général"),
                                          contexte=contexte_recherche)
       sys_content += extraits_pertinents(st.session_state.current_chat_id, prompt_courant)

   api = [{"role": "system", "content": sys_content}]
   resume = obtenir_resume(chat)
   if resume:
       api.append({"role": "system", "content": f"# RÉSUMÉ DES ÉCHANGES PRÉCÉDENTS\n{resume}"})
   fenetre = chat["messages"][-MAX_CONTEXT_MESSAGES:]
   for m in fenetre:
       est_dernier = (m is chat["messages"][-1])
       if est_dernier and m.get("image_b64"):
           mime = m.get("image_mime", "image/jpeg")
           api.append({"role": "user", "content": [
               {"type": "text", "text": m["content"]},
               {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{m['image_b64']}"}}]})
       elif m.get("document"):
           api.append({"role": m["role"],
                       "content": f"{m['content']}\n\n[Document joint]\n{m['document'][:12000]}"})
       else:
           api.append({"role": m["role"], "content": m["content"]})
   return api

def generer_titre_automatique(chat):
   if chat.get("titre_manuel"): return
   echanges = [m for m in chat["messages"] if m["role"] in ("user", "assistant")]
   if len(echanges) < 2: return
   try:
       r = client.chat.completions.create(
           model=MODEL_LIGHT,
           messages=[{"role": "system", "content":
                      "Donne un titre à cette conversation : 3 à 6 mots, français, sans guillemets ni ponctuation finale."},
                     {"role": "user", "content": "\n".join(str(m["content"])[:300] for m in echanges[:4])}],
           max_tokens=20, temperature=0.3)
       titre = r.choices[0].message.content.strip().strip('"').strip("'")[:50]
       if titre: chat["title"] = titre
   except Exception: pass
   if chat["title"] == "Nouvelle discussion":
       premier = next((str(m["content"]) for m in chat["messages"] if m["role"] == "user"), "")
       chat["title"] = (premier[:36] + "…") if len(premier) > 36 else (premier or "Discussion")

# ================= 16. AUDIO, STREAMING & FALLBACK =================
def nettoyer_pour_audio(t):
   t = re.sub(r"```.*?```", " (bloc de code) ", t, flags=re.DOTALL)
   t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
   return re.sub(r"[#*>`]", "", t)[:4500]

@st.cache_data(ttl=86400 * 7, max_entries=200, show_spinner=False)
def generer_audio_cached(empreinte, texte, voice_id):
   if eleven_client is None: return None
   try:
       flux = eleven_client.text_to_speech.convert(
           voice_id=voice_id, output_format="mp3_44100_128",
           text=texte, model_id="eleven_multilingual_v2")
       return b"".join(chunk for chunk in flux)
   except Exception: return None

def generer_audio(texte):
   propre = nettoyer_pour_audio(texte)
   empreinte = hashlib.md5((propre + VOICE_ID).encode()).hexdigest()
   return generer_audio_cached(empreinte, propre, VOICE_ID)

def stream_edith(box, candidats, api_messages):
   derniere_erreur = None
   for i, mdl in enumerate(candidats):
       try:
           if i > 0:
               box.empty()
               st.toast(f"Bascule de secours → {mdl.split('/')[-1].replace(':online','')}", icon="🛟")
           holder = {"usage": None}
           stream = client.chat.completions.create(
               model=mdl, messages=api_messages, stream=True,
               stream_options={"include_usage": True},
               temperature=st.session_state.temperature,
               max_tokens=st.session_state.max_tokens)
           acc = ""
           for chunk in stream:
               if getattr(chunk, "usage", None): holder["usage"] = chunk.usage
               if chunk.choices and chunk.choices[0].delta.content:
                   acc += chunk.choices[0].delta.content
                   box.markdown(acc + " ▌")
           if not acc.strip(): raise RuntimeError("Réponse vide du modèle")
           return acc, holder["usage"], mdl, (i > 0)
       except Exception as e: derniere_erreur = e
   raise derniere_erreur

MOTS_REFUS = ["je ne peux pas vous aider", "i cannot assist", "i can't assist",
             "i cannot help with", "against my guidelines", "against my programming"]
def est_un_refus(texte):
   return any(m in texte.lower() for m in MOTS_REFUS)

def bulle_user(texte):
   st.markdown(f'<div class="msg-row user"><div class="msg-bubble">{html.escape(texte)}</div></div>', unsafe_allow_html=True)

def tete_edith():
   st.markdown('<div class="msg-row edith"><div class="edith-avatar">⚡</div><div class="edith-nom">E.D.I.T.H.</div></div>', unsafe_allow_html=True)

def ligne_statut(meta):
   modele = html.escape(str(meta.get("model", "?")).replace(":online", ""))
   raison = html.escape(str(meta.get("reason", "")))
   ligne = f"<code>{modele}</code> · {raison}"
   if meta.get("web"): ligne += " · 🌐 web"
   if st.session_state.get("show_debug", False) and PROFIL == "arthur":
       ligne += f" · in:{meta.get('tokens_in','?')} / out:{meta.get('tokens_out','?')} tok"
   st.markdown(f'<div class="statut">{ligne}</div>', unsafe_allow_html=True)

# ================= 17. SIDEBAR (ordre façon ChatGPT) =================
mode_choisi = "🤖 Automatique (Routeur)"
selected_manual_model = None
search_query = ""

with st.sidebar:
   st.markdown('<div class="brand-title">E.D.I.T.H.</div>', unsafe_allow_html=True)
   sous_titre = "ÉDITION PADRE" if PROFIL == "padre" else "EVEN DEAD I'M THE HERO"
   st.markdown(f'<div class="brand-sub">{sous_titre}</div>', unsafe_allow_html=True)

   # --- 1. Nouveau chat ---
   if st.button("➕ Nouveau chat", type="primary", use_container_width=True):
       st.session_state.incognito = False
       create_new_chat()
       st.rerun()

   # --- 2. Rechercher (filtre l'historique) ---
   search_query = st.text_input("🔍 Rechercher…", label_visibility="collapsed",
                                placeholder="🔍 Rechercher…").strip()

   if not st.session_state.incognito:
       # --- 3. Projets ---
       st.markdown('<div class="side-label">Projets</div>', unsafe_allow_html=True)
       projets = liste_projets()

       # --- FIX STREAMLIT SELECTBOX (conservé) ---
       if st.session_state.get("projet_courant") not in projets:
           st.session_state["projet_courant"] = "Général"

       index_projet = projets.index(st.session_state["projet_courant"])

       col_proj, col_add = st.columns([0.78, 0.22])
       with col_proj:
           projet_selectionne = st.selectbox("Projet actif", projets, index=index_projet, label_visibility="collapsed")
           if projet_selectionne != st.session_state["projet_courant"]:
               st.session_state["projet_courant"] = projet_selectionne
               st.rerun()

       with col_add:
           with st.popover("➕"):
               nom_projet = st.text_input("Nom du projet :", key="input_nouveau_projet", placeholder="Voiture RC, Drone…")
               if st.button("Créer le projet", use_container_width=True):
                   if nom_projet.strip():
                       ajouter_projet(nom_projet.strip())
                       st.session_state["projet_courant"] = nom_projet.strip()
                       st.rerun()

       # --- 4. Récents (chats du projet courant) ---
       st.markdown('<div class="side-label">Récents</div>', unsafe_allow_html=True)

       for c_id in reversed(list(st.session_state.chats.keys())):
           data = st.session_state.chats[c_id]
           if data.get("projet", "Général") != st.session_state["projet_courant"]: continue
           if not chat_matches_search(data, search_query): continue

           is_active = (c_id == st.session_state.current_chat_id)
           label_titre = ("📌 " if is_active else "💭 ") + data["title"]
           col_title, col_menu = st.columns([0.84, 0.16])
           with col_title:
               if st.button(label_titre, key=f"btn_chat_{c_id}", use_container_width=True):
                   st.session_state.current_chat_id = c_id
                   st.session_state["projet_courant"] = data.get("projet", "Général")
                   st.rerun()
           with col_menu:
               with st.popover("⋮"):
                   st.markdown("**Options du chat**")
                   nouveau_titre = st.text_input("Titre :", value=data["title"], key=f"rename_input_{c_id}")
                   if st.button("✏️ Modifier le titre", key=f"rename_btn_{c_id}", use_container_width=True):
                       if nouveau_titre.strip():
                           data["title"] = nouveau_titre.strip(); data["titre_manuel"] = True; sauvegarder_historique(); st.rerun()
                   st.download_button("⬇️ Exporter (.md)", data=exporter_markdown(data), file_name=re.sub(r"[^\w\- ]", "", data["title"])[:30] + ".md", mime="text/markdown", key=f"export_{c_id}", use_container_width=True)
                   st.divider()
                   confirmation = st.checkbox("Confirmer la suppression", key=f"confirm_del_{c_id}")
                   if st.button("🗑️ Supprimer chat & mémoires", key=f"del_chat_{c_id}", use_container_width=True, disabled=not confirmation):
                       supprimer_memoires_discussion(c_id); del st.session_state.chats[c_id]; sauvegarder_historique()
                       if not st.session_state.chats: create_new_chat()
                       else: st.session_state.current_chat_id = next(reversed(st.session_state.chats))
                       st.rerun()

   # --- 5. Outils ---
   with st.expander("🧰 Outils"):
       if st.button("☀️ Briefing du matin", use_container_width=True):
           st.session_state.pending_prompt = (
               "Bonjour E.D.I.T.H. Fais-moi le briefing du matin : date et heure, rappels du jour et à venir, "
               "puis les souvenirs récents qui méritent mon attention.")
           st.rerun()

       st.toggle("🕶️ Mode incognito", key="incognito",
                 help="Rien n'est sauvegardé : ni historique, ni souvenirs, ni rappels.")
       if st.session_state.incognito:
           if st.button("🧹 Effacer la session incognito", use_container_width=True):
               st.session_state.chat_incognito = {"title": "Session incognito", "messages": [], "projet": "Général"}
               st.rerun()
       st.toggle("🔊 Mode Vocal Continu", key="mode_vocal_continu",
                 help="Diffuse automatiquement chaque nouvelle réponse en audio.")
       st.toggle("🌐 Recherche web", key="web_actif",
                 help="Connecte E.D.I.T.H. à Internet via OpenRouter (:online). Petit coût par requête.")

       st.markdown('<div class="side-label">Modèle</div>', unsafe_allow_html=True)
       mode_choisi = st.radio("Sélection du modèle", ["🤖 Automatique (Routeur)", "🎛️ Manuel"],
                              label_visibility="collapsed", key="mode_radio")
       if mode_choisi == "🎛️ Manuel":
           selected_manual_model = MODELS_PROFIL[st.selectbox("Choisir l'IA :", list(MODELS_PROFIL.keys()), label_visibility="collapsed")]

       st.markdown('<div class="side-label">Rappels</div>', unsafe_allow_html=True)
       with st.popover("➕ Nouveau rappel", use_container_width=True):
           r_date = st.date_input("Date :", value=datetime.now().date(), key="rappel_date")
           avec_heure = st.checkbox("Préciser une heure", key="rappel_avec_heure")
           r_heure = st.time_input("Heure :", key="rappel_heure").strftime("%H:%M") if avec_heure else ""
           r_msg = st.text_input("Message :", key="rappel_msg", placeholder="Appeler le garage…")
           if st.button("Ajouter le rappel", use_container_width=True):
               if r_msg.strip():
                   ajouter_rappel(r_date.isoformat(), r_heure, r_msg.strip()); st.toast("Rappel ajouté ⏰"); st.rerun()
       rappels_a_venir = sorted([r for r in charger_rappels() if not r.get("fait")], key=lambda x: (x["date"], x.get("heure", "")))[:6]
       for r in rappels_a_venir:
           en_retard = r["date"] <= date_iso()
           c_txt, c_del = st.columns([0.85, 0.15])
           with c_txt: st.caption(f"{'🔴' if en_retard else '⚪'} {r['date'][8:10]}/{r['date'][5:7]} {r.get('heure','')} — {r['message'][:38]}")
           with c_del:
               if st.button("🗑️", key=f"del_rappel_{r['id']}"): supprimer_rappel(r["id"]); st.rerun()

   # --- 6. Mémoire ---
   if memoire_collection is not None:
       with st.expander("🧠 Mémoire"):
           nb = memoire_collection.count()
           st.markdown(f'<div class="mem-count">🧠 {nb} souvenir{"s" if nb > 1 else ""} dans la mémoire absolue</div>', unsafe_allow_html=True)
           if PROFIL == "arthur":
               with st.form("form_ajout_memoire", clear_on_submit=True):
                   nouvelle_info = st.text_input("Ajouter un souvenir :", placeholder="ex: Arthur aime coder tard la nuit")
                   if st.form_submit_button("➕ Ajouter", use_container_width=True):
                       if nouvelle_info.strip():
                           fr, cree = sauvegarder_souvenir(nouvelle_info.strip(), manuel=True, projet=st.session_state.get("projet_courant", "Général"))
                           st.toast(f"Souvenir ajouté le {fr} !" if cree else "Doublon ignoré.", icon="🧠"); st.rerun()
               st.markdown("**Derniers souvenirs :**")
               tous = memoire_collection.get(limit=50)
               if tous and tous.get("ids"):
                   for s_id, s_doc in zip(reversed(tous["ids"]), reversed(tous["documents"])):
                       c_text, c_del = st.columns([0.82, 0.18])
                       with c_text: st.caption(s_doc)
                       with c_del:
                           if st.session_state.get("confirm_mem") == s_id:
                               if st.button("✅", key=f"cfm_mem_{s_id}"):
                                   memoire_collection.delete(ids=[s_id]); st.session_state.pop("confirm_mem", None); backup_memoire(); st.toast("Souvenir supprimé !", icon="🗑️"); st.rerun()
                           else:
                               if st.button("🗑️", key=f"del_mem_{s_id}"): st.session_state.confirm_mem = s_id; st.rerun()
               else: st.caption("Aucun souvenir dans la mémoire.")
               # Sauvegarde JSON téléchargeable (restaurée automatiquement au boot)
               backup_data = lire_json(FICHIER_BACKUP_MEMOIRE, [])
               st.download_button("⬇️ Sauvegarde mémoire (.json)",
                                  data=json.dumps(backup_data, ensure_ascii=False, indent=2),
                                  file_name=f"backup_memoire_{PROFIL}.json",
                                  mime="application/json", key="dl_backup_memoire",
                                  use_container_width=True)

   # --- 7. Réglages (Arthur uniquement) ---
   if PROFIL == "arthur":
       with st.expander("⚙️ Réglages"):
           st.markdown('<div class="side-label">Budget du mois</div>', unsafe_allow_html=True)
           dep = lire_json(FICHIER_DEPENSES, {})
           m = dep.get(datetime.now().strftime("%Y-%m"), {"tokens_in": 0, "tokens_out": 0, "cout_usd": 0.0})
           cout_eur = m["cout_usd"] * TAUX_USD_EUR
           st.caption(f"Tokens : {m['tokens_in']:,} entrée / {m['tokens_out']:,} sortie")
           st.caption(f"Coût estimé : **{cout_eur:.2f} €** (indicatif, prix OpenRouter live)")
           if cout_eur > BUDGET_ALERTE: st.warning(f"Seuil d'alerte dépassé ({BUDGET_ALERTE:.0f} €) !")

           st.markdown('<div class="side-label">Atelier</div>', unsafe_allow_html=True)
           st.slider("Température", 0.0, 1.5, key="temperature", step=0.05)
           st.slider("Longueur max (tokens)", 256, 32768, key="max_tokens", step=256)

           st.toggle("Debug sous le capot", key="show_debug")
           st.toggle("Reroutage anti-refus en auto", key="reroutage_refus")
           # Garde-fou : une valeur V2 persistée (ex. 1.35) serait hors des nouvelles bornes
           st.session_state["seuil_memoire"] = min(
               max(st.session_state.get("seuil_memoire", SEUIL_PERTINENCE_DEFAUT), 0.30), 0.90)
           st.slider("Seuil de pertinence mémoire", 0.30, 0.90, key="seuil_memoire", step=0.02,
                     help="Distance cosine maximale pour restituer un souvenir (plus bas = plus strict).")

           st.markdown('<div class="side-label">Catalogue OpenRouter</div>', unsafe_allow_html=True)
           if st.button("📡 Lister les modèles OpenRouter", use_container_width=True):
               if REQUESTS_DISPONIBLE:
                   try:
                       r = requests.get("https://openrouter.ai/api/v1/models", timeout=10)
                       liste = []
                       for mod in r.json().get("data", []):
                           p = mod.get("pricing", {})
                           liste.append(f"{mod['id']} · ctx {mod.get('context_length','?')} · "
                                        f"${float(p.get('prompt') or 0)*1e6:.2f}/${float(p.get('completion') or 0)*1e6:.2f} par M")
                       st.session_state.modeles_or = sorted(liste)
                   except Exception as e: st.error(f"Catalogue inaccessible : {e}")
           if st.session_state.get("modeles_or"):
               filtre = st.text_input("Filtrer :", key="filtre_modeles", placeholder="claude, gpt, kimi…")
               for ligne in st.session_state.modeles_or:
                   if not filtre or filtre.lower() in ligne.lower(): st.caption(ligne)

   # --- 8. Profil + verrouillage (bas de sidebar) ---
   st.markdown("---")
   nom_profil = "Arthur" if PROFIL == "arthur" else "Padre"
   st.markdown(f'<div class="profil-chip">👤 {nom_profil}</div>', unsafe_allow_html=True)
   if st.button("🔒 Verrouiller", use_container_width=True):
       for k in list(st.session_state.keys()): del st.session_state[k]
       st.rerun()

# ================= 18. ZONE PRINCIPALE =================
chat = chat_courant()
messages = chat["messages"]
incognito = st.session_state.get("incognito", False)

# Rappels dus (affichés en haut, jamais en incognito)
if not incognito:
   dus = [r for r in charger_rappels() if not r.get("fait") and r["date"] <= date_iso()]
   if dus:
       for r in dus:
           c_txt, c_ok = st.columns([0.88, 0.12])
           with c_txt: st.warning(f"⏰ **Rappel** : {r['message']} (prévu le {r['date'][8:10]}/{r['date'][5:7]} {r.get('heure','')})")
           with c_ok:
               if st.button("✅ Fait", key=f"fait_{r['id']}"): marquer_rappel_fait(r["id"]); st.rerun()

mode_label = "🤖 Routeur" if mode_choisi == "🤖 Automatique (Routeur)" else f"🎛️ {selected_manual_model.split('/')[-1]}"
pills = f'<span class="pill">{html.escape(mode_label)}</span>'
if chat.get("projet", "Général") != "Général" and not incognito: pills += f'<span class="pill">📁 {html.escape(chat["projet"])}</span>'
if st.session_state.web_actif: pills += '<span class="pill">🌐 Web</span>'
if incognito: pills += '<span class="pill">🕶️ Incognito</span>'
pills += f'<span class="pill">📅 {date_complete()}</span>'
st.markdown(f'<div class="topbar"><span class="chat-title">{html.escape(chat["title"])}</span><span>{pills}</span></div>', unsafe_allow_html=True)
st.divider()

# Hero centré quand le chat est vide (variante incognito conservée)
if not messages:
   if incognito:
       st.markdown("""<div class="hero"><h1>🕶️ Session incognito.</h1><p>Rien ne sera conservé. Même moi, je ferai semblant d'oublier.</p></div>""", unsafe_allow_html=True)
   else:
       st.markdown("""<div class="hero"><h1>Quel est le programme aujourd'hui ?</h1><p>E.D.I.T.H. est prête. Posez votre question, joignez une image ou un document.</p></div>""", unsafe_allow_html=True)

for idx, m in enumerate(messages):
   if m["role"] == "user":
       bulle_user(m["content"])
       if "image_b64" in m: st.image(base64.b64decode(m["image_b64"]), width=280)
       if m.get("document_nom"): st.caption(f"📎 {m['document_nom']}")
   else:
       tete_edith()
       st.markdown(m["content"])
       if "metadata" in m: ligne_statut(m["metadata"])
       col_audio, _ = st.columns([0.15, 0.85])
       with col_audio:
           with st.popover("🔊 Écouter"):
               if st.button("▶️ Lire ce message", key=f"play_single_{idx}"):
                   audio_bytes = generer_audio(m["content"])
                   if audio_bytes: st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                   else: st.error("Impossible de générer l'audio.")

# Régénération multi-modèle (dernier message assistant)
if messages and messages[-1]["role"] == "assistant" and messages[-1].get("content"):
   with st.popover("🔁 Régénérer cette réponse"):
       options = ["🤖 Routeur"] + list(MODELS_PROFIL.keys())
       choix_regen = st.selectbox("Avec quel modèle ?", options, key="regen_choix")
       if st.button("🔁 Régénérer", use_container_width=True, key="btn_regen"):
           messages.pop(); st.session_state.regen_demandee = True
           st.session_state.regen_model = None if choix_regen.startswith("🤖") else MODELS_PROFIL[choix_regen]
           if not incognito: sauvegarder_historique()
           st.rerun()

# ================= 19. ENTRÉES & TRAITEMENT =================
# Les 3 entrées (image / micro / document) regroupées dans un popover « ＋ Joindre »
suffixe_up = f"{st.session_state.current_chat_id}_{st.session_state.upload_counter}"
with st.popover("＋ Joindre"):
   uploaded_image = st.file_uploader("🖼️ Image", type=["png", "jpg", "jpeg", "webp"], key=f"uploader_img_{suffixe_up}")
   audio_val = st.audio_input("🎤 Parler à E.D.I.T.H.", key=f"audio_in_{suffixe_up}")
   types_doc = ["txt", "md"] + (["pdf"] if PDF_DISPONIBLE else [])
   uploaded_doc = st.file_uploader("📎 Document", type=types_doc, key=f"uploader_doc_{suffixe_up}", help="PDF, TXT ou MD. Petit fichier : injecté directement. Gros : indexé pour recherche.")

prompt = st.chat_input("Demandez quoi que ce soit à E.D.I.T.H.…")
if "pending_prompt" in st.session_state: prompt = st.session_state.pop("pending_prompt")

regen_actif = False
if prompt is None and st.session_state.get("regen_demandee") and messages and messages[-1]["role"] == "user":
   regen_actif = True

if audio_val is not None:
   if st.session_state.get("last_audio_id") != audio_val.id:
       st.session_state.last_audio_id = audio_val.id
       if openai_client:
           with st.spinner("Transcription en cours (Whisper)..."):
               try:
                   transcript = openai_client.audio.transcriptions.create(model="whisper-1", file=("audio.wav", audio_val))
                   prompt = transcript.text or prompt
               except Exception as e: st.error(f"Erreur de transcription : {e}")
       else: st.error("Clé OpenAI manquante pour utiliser le micro.")

if (prompt and prompt.strip()) or regen_actif:
   if regen_actif:
       user_msg = messages[-1]
       prompt_text = user_msg["content"]
   else:
       prompt_text = prompt.strip()
       user_msg = {"role": "user", "content": prompt_text}
       if uploaded_image:
           user_msg["image_b64"] = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
           user_msg["image_mime"] = uploaded_image.type or "image/jpeg"
       if uploaded_doc:
           texte_doc = extraire_texte_fichier(uploaded_doc)
           if texte_doc is None: st.error("Lecture du document impossible (pypdf est-il installé ?).")
           elif not texte_doc.strip(): st.warning("Document vide ou texte non extractible (PDF scanné ?).")
           else:
               user_msg["document_nom"] = uploaded_doc.name
               if len(texte_doc) <= 12000: user_msg["document"] = texte_doc
               else:
                   nb_chunks = indexer_document(st.session_state.current_chat_id, uploaded_doc.name, texte_doc)
                   user_msg["document"] = (f"[Document « {uploaded_doc.name} » trop long pour injection directe : "
                                           f"{nb_chunks} extraits indexés, recherchez dedans via les questions.]"
                                           if nb_chunks else "[Échec d'indexation du document.]")
       messages.append(user_msg)
       if not incognito: sauvegarder_historique()
       bulle_user(prompt_text)
       if uploaded_image: st.image(uploaded_image, width=280)
       if user_msg.get("document_nom"): st.caption(f"📎 {user_msg['document_nom']}")

   tete_edith()
   box = st.empty()
   with st.spinner("Analyse de la demande…"):
       if regen_actif and st.session_state.get("regen_model"):
           selected_model, route_reason = st.session_state.regen_model, "Régénération manuelle"
       elif regen_actif:
           selected_model, route_reason = get_smart_route(prompt_text, user_msg.get("image_b64") is not None)
           route_reason = "Régénération · " + route_reason
       elif mode_choisi == "🎛️ Manuel":
           selected_model, route_reason = selected_manual_model, "Sélection manuelle"
           if user_msg.get("image_b64") and selected_model not in MODELES_VISION:
               route_reason = "Modèle sans vision → bascule vision"
               selected_model = MODEL_LIGHT
       else:
           contexte_routeur = " / ".join(str(m["content"])[:120] for m in messages[-3:-1]) if len(messages) > 1 else ""
           selected_model, route_reason = get_smart_route(prompt_text, user_msg.get("image_b64") is not None, contexte_routeur)

   web = st.session_state.get("web_actif", False)
   def avec_online(m): return m + ":online" if web else m
   candidats = [avec_online(selected_model)] + [avec_online(m) for m in FALLBACKS if m != selected_model]

   api_messages = construire_api_messages(chat, prompt_text)

   try:
       # NOUVELLE PARTIE : On bypass complétement l'IA si la commande spéciale [DUMP_MEMORY] est incluse
       if "[DUMP_MEMORY]" in prompt_text:
           if memoire_collection is None or memoire_collection.count() == 0:
               texte = "Ma mémoire vectorielle est actuellement vide."
           else:
               tous = memoire_collection.get()
               lignes = []
               for m, d in zip(tous.get("metadatas", []), tous.get("documents", [])):
                   date_s = m.get("date_fr", "?")
                   lignes.append(f"- [{date_s}] {d}")
               texte = "🧠 **DUMP COMPLET DE LA MÉMOIRE :**\n\n" + "\n".join(lignes)
           
           usage = None
           modele_utilise = "Système (Interne)"
           route_reason = "Commande secrète [DUMP_MEMORY]"
           a_fallback = False
       else:
           # Comportement normal si la balise n'y est pas
           texte, usage, modele_utilise, a_fallback = stream_edith(box, candidats, api_messages)

           # === LOGIQUE ANTI-REFUS (reroutage optionnel, debuggable) ===
           if (PROFIL == "arthur" and st.session_state.get("reroutage_refus")
                   and est_un_refus(texte) and modele_utilise.replace(":online", "") != MODEL_UNFILTERED
                   and mode_choisi == "🤖 Automatique (Routeur)"):

               if st.session_state.get("show_debug", False):
                   with st.expander("⚠️ DÉBOGAGE : Réponse censurée (Faux positif ?)"):
                       st.write(texte)

               st.toast("Refus détecté — bascule sur Grok 4.3", icon="🛡️")
               route_reason += " ➔ reroutage post-refus"
               box.empty()
               texte, usage, modele_utilise, _ = stream_edith(box, [avec_online(MODEL_UNFILTERED)], api_messages)

       if a_fallback: route_reason += " (après secours)"

       texte_propre = texte
       saves = re.findall(r"\[SAVE:\s*(.*?)\]", texte, re.IGNORECASE)
       if saves:
           if incognito: st.toast("Mode incognito : souvenir non gravé.", icon="🕶️")
           elif memoire_collection is not None:
               for info in saves:
                   fr, cree = sauvegarder_souvenir(info.strip(), chat_id=st.session_state.current_chat_id, projet=chat.get("projet", "Général"))
                   if cree: st.toast(f"Souvenir gravé le {fr} : {info.strip()}", icon="🧠")
           texte_propre = re.sub(r"\s*\[SAVE:\s*.*?\]", "", texte_propre, flags=re.IGNORECASE)

       rappels_detectes = re.findall(r"\[RAPPEL:\s*(\d{4}-\d{2}-\d{2})(?:\s+(\d{1,2}:\d{2}))?\s*\|\s*([^\]]+)\]", texte_propre, re.IGNORECASE)
       if rappels_detectes:
           if incognito: st.toast("Mode incognito : rappel non enregistré.", icon="🕶️")
           else:
               for d_r, h_r, msg_r in rappels_detectes:
                   ajouter_rappel(d_r, h_r or "", msg_r.strip())
                   st.toast(f"Rappel posé pour le {d_r} {h_r or ''} : {msg_r.strip()}", icon="⏰")
               texte_propre = re.sub(r"\s*\[RAPPEL:[^\]]*\]", "", texte_propre, flags=re.IGNORECASE)

       texte_propre = texte_propre.strip()
       box.markdown(texte_propre)

       if st.session_state.get("mode_vocal_continu", False):
           audio_bytes = generer_audio(texte_propre)
           if audio_bytes: st.audio(audio_bytes, format="audio/mp3", autoplay=True)

       meta = {"model": modele_utilise, "reason": route_reason, "web": web,
               "tokens_in": getattr(usage, "prompt_tokens", "?") if usage else "?",
               "tokens_out": getattr(usage, "completion_tokens", "?") if usage else "?"}
       ligne_statut(meta)
       messages.append({"role": "assistant", "content": texte_propre, "metadata": meta})

       enregistrer_depense(modele_utilise, usage, web_utilise=web)

       if not incognito:
           nb_echanges = len([m for m in messages if m["role"] in ("user", "assistant")])
           if not chat.get("titre_manuel") and (chat["title"] == "Nouvelle discussion" or nb_echanges <= 3):
               generer_titre_automatique(chat)
           sauvegarder_historique()

       st.session_state.pop("regen_demandee", None)
       st.session_state.pop("regen_model", None)
       st.session_state.upload_counter += 1
       st.rerun()

   except Exception as e:
       st.session_state.pop("regen_demandee", None)
       st.session_state.pop("regen_model", None)
       box.error(f"Erreur d'exécution de l'IA : {e}")
       if st.session_state.get("show_debug", False) and PROFIL == "arthur":
           st.code(traceback.format_exc(), language="python")
