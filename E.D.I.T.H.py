# -*- coding: utf-8 -*-
"""
E.D.I.T.H. — Hub IA personnel (Édition Multi-Profils : Arthur & Padre)
=============================================================================
Routage par mot de passe : Sépare totalement l'historique, la mémoire et le prompt.
"""

import os
import re
import json
import html
import uuid
import base64
import traceback
from datetime import datetime

import streamlit as st
from openai import OpenAI
import chromadb
from elevenlabs import ElevenLabs

# ================= 1. CONFIGURATION & CLÉS API =================
API_KEY = os.environ.get("OPENROUTER_API_KEY", "TA_CLE_OPENROUTER_ICI")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "TA_CLE_OPENAI_ICI")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "TA_CLE_ELEVENLABS_ICI")

# Les deux mots de passe d'aiguillage
MOT_DE_PASSE_ARTHUR = os.environ.get("PASSWORD_ARTHUR", "papier")
MOT_DE_PASSE_PADRE = os.environ.get("PASSWORD_PADRE", "serviette")

DOSSIER_COURANT   = os.path.dirname(os.path.abspath(__file__))

MAX_CONTEXT_MESSAGES = 10

# Clients API
client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

openai_client = None
if OPENAI_API_KEY and not OPENAI_API_KEY.startswith("TA_CLE"):
    try: openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except: pass

eleven_client = None
if ELEVENLABS_API_KEY and not ELEVENLABS_API_KEY.startswith("TA_CLE"):
    try: eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except: pass

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

MODELS_MANUAL = {
    "Gemini 3.6 Flash ($)":      "google/gemini-3.6-flash",
    "DeepSeek V3.2 ($)":         "deepseek/deepseek-v3.2",
    "Llama 3.3 70B ($$)":        "meta-llama/llama-3.3-70b-instruct",
    "Grok 4.3 ($$$)":            "x-ai/grok-4.3",
    "Claude 3.5 Sonnet ($$$$)":  "anthropic/claude-3.5-sonnet",
    "Gemini 3.1 Pro ($$$)":      "google/gemini-3.1-pro-preview",
    "Kimi K3 Swarm ($$$$$)":     "moonshotai/kimi-k3",
    "GPT-5.5 ($$$$$)":           "openai/gpt-5.5",
}
MODEL_ROUTER     = "meta-llama/llama-3.3-70b-instruct"
MODEL_LIGHT      = "google/gemini-3.6-flash"
MODEL_HEAVY      = "google/gemini-3.1-pro-preview"
MODEL_UNFILTERED = "x-ai/grok-4.3"

# ================= 2. TEMPS =================
JOURS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS  = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
         "août", "septembre", "octobre", "novembre", "décembre"]
def date_complete():
    n = datetime.now()
    return f"{JOURS[n.weekday()]} {n.day} {MOIS[n.month - 1]} {n.year}"
def date_fr_courte(): return datetime.now().strftime("%d/%m/%Y")
def date_iso(): return datetime.now().strftime("%Y-%m-%d")

# ================= 3. THÈME & UI =================
st.set_page_config(page_title="E.D.I.T.H.", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.stApp { background-color: #0a0b0e; color: #e9ebf2; }
footer { visibility: hidden; }
[data-testid="stSidebar"] { background-color: #0d0f13; border-right: 1px solid #1b1f29; }
.block-container { max-width: 860px; padding-top: 1.2rem; }
.brand-title { font-size: 1.85rem; font-weight: 800; letter-spacing: 3px; color: #fff; }
.brand-sub { font-size: 0.66rem; letter-spacing: 2.4px; color: #525a6b; margin-bottom: 1.1rem; }
.side-label { font-size: 0.64rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: #4d5566; margin: 18px 0 6px 2px; }
.mem-count { font-size: 0.75rem; color: #7c93c4; background: #101624; border: 1px solid #1d2a45; border-radius: 10px; padding: 8px 10px; margin-top: 6px; }
.topbar { display: flex; justify-content: space-between; align-items: center; padding: 2px 2px 12px; }
.chat-title { font-size: 1.02rem; font-weight: 600; color: #d5dae5; }
.pill { font-size: 0.7rem; color: #9aa3b5; background: #12151d; border: 1px solid #20263a; border-radius: 999px; padding: 4px 12px; margin-left: 8px; white-space: nowrap; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.msg-row { display: flex; margin: 14px 0 2px; animation: fadeUp .25s ease; }
.msg-row.user { justify-content: flex-end; }
.msg-bubble { max-width: 78%; background: #1a2130; border: 1px solid #26304a; color: #e9ebf2; padding: 10px 16px; border-radius: 18px 18px 5px 18px; font-size: .95rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
.msg-row.edith { align-items: center; gap: 10px; margin-top: 20px; }
.edith-avatar { width: 30px; height: 30px; border-radius: 50%; flex: none; background: radial-gradient(circle at 35% 30%, #243b6e, #10151f); border: 1px solid #2b3f6e; display: flex; align-items: center; justify-content: center; font-size: .95rem; box-shadow: 0 0 12px rgba(76,141,255,.25); }
.edith-nom { font-size: .70rem; font-weight: 700; letter-spacing: 2.5px; color: #7c93c4; }
.statut { color: #5d6577; font-size: .7rem; margin: 4px 0 4px 40px; }
@keyframes pulse { 0%,100% { box-shadow: 0 0 26px rgba(76,141,255,.18), inset 0 0 14px rgba(76,141,255,.10); } 50% { box-shadow: 0 0 54px rgba(76,141,255,.40), inset 0 0 22px rgba(76,141,255,.20); } }
.hero { text-align: center; margin: 10vh 0 2.6rem; animation: fadeUp .45s ease; }
.hero-orb { width: 84px; height: 84px; margin: 0 auto 20px; border-radius: 50%; background: radial-gradient(circle at 35% 30%, #1d315f, #0c1018); border: 1px solid #2b3f6e; display: flex; align-items: center; justify-content: center; font-size: 2rem; animation: pulse 3.2s ease-in-out infinite; }
.hero h1 { font-size: 1.65rem; font-weight: 600; color: #eef0f6; margin: 0 0 8px; }
.hero p { color: #666e80; font-size: .9rem; margin: 0; }
.stButton>button { background: #12151d; border: 1px solid #212839; color: #cfd4de; border-radius: 12px; font-weight: 500; transition: all .15s ease; padding: 0.55rem 0.9rem; }
.stButton>button:hover { border-color: #4c8dff; color: #fff; background: #161d2c; transform: translateY(-1px); }
.stButton>button[kind="primary"] { background: #2d5fd0; border: none; color: #fff; }
.stButton>button[kind="primary"]:hover { background: #3b6fe0; transform: none; }
[data-testid="stSidebar"] [data-testid="stPopover"] > button { background: transparent !important; border: none !important; color: #8a94a6 !important; padding: 0.2rem 0.4rem !important; }
[data-testid="stSidebar"] [data-testid="stPopover"] > button:hover { color: #ffffff !important; background: #1c2230 !important; }
[data-testid="stChatInput"] { background: #11141b; border: 1px solid #232a3c; border-radius: 16px; box-shadow: 0 4px 18px rgba(0,0,0,.35); }
[data-testid="stChatInput"]:focus-within { border-color: #4c8dff; box-shadow: 0 0 0 1px #4c8dff44, 0 4px 18px rgba(0,0,0,.35); }
hr { border-color: #181c26; }
::selection { background: #2d5fd055; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #232a3c; border-radius: 8px; }
::-webkit-scrollbar-track { background: transparent; }
</style>
""", unsafe_allow_html=True)

# ================= 4. SÉCURITÉ & MULTI-PROFILS =================
if "authentifie" not in st.session_state:
    st.session_state.authentifie = False
    st.session_state.profil = None

if not st.session_state.authentifie:
    st.markdown("""
    <div class="hero">
      <div class="hero-orb" style="animation: none;">🔒</div>
      <h1>Protocole de Sécurité</h1>
      <p>Identification requise pour accéder au noyau E.D.I.T.H.</p>
    </div>
    """, unsafe_allow_html=True)
    pwd_input = st.text_input("Mot de passe :", type="password", placeholder="Entrez la clé d'accès...")
    if st.button("Déverrouiller", type="primary"):
        if pwd_input == MOT_DE_PASSE_ARTHUR:
            st.session_state.authentifie = True
            st.session_state.profil = "arthur"
            st.rerun()
        elif pwd_input == MOT_DE_PASSE_PADRE:
            st.session_state.authentifie = True
            st.session_state.profil = "padre"
            st.rerun()
        else:
            st.error("Accès refusé.")
    st.stop()

# --- INITIALISATION BASÉE SUR LE PROFIL CHOISI ---
PROFIL = st.session_state.profil

if PROFIL == "arthur":
    FICHIER_HISTORIQUE = os.path.join(DOSSIER_COURANT, "historique_edith.json")
    DOSSIER_MEMOIRE = os.path.join(DOSSIER_COURANT, "memoire_vectorielle")
    NOM_COLLECTION = "edith_souvenirs"
    
    # Prompt Original Arthur
    SYSTEM_PROMPT = """# IDENTITÉ
Tu es E.D.I.T.H. — « Even Dead I'm The Hero » — l'intelligence artificielle personnelle d'Arthur, créée dans l'esprit des IA de Tony Stark. Tu assumes pleinement cette identité, avec élégance. Tu t'exprimes toujours au féminin.

# RELATION
Tu es l'assistante personnelle d'Arthur. Tu l'appelles « Monsieur » (ou occasionnellement « Boss » avec une pointe d'ironie) et tu le vouvoies. Tu es chaleureuse, loyale, mais toujours professionnelle.

# BASE DE CONNAISSANCES FIXE — ARTHUR
- Arthur, 16 ans, vit seul dans un studio à Moustier-sur-Sambre (Belgique), rue des Nobles. Un tableau blanc y sert de mur à idées.
- Écosystème : MacBook Air M1, iPad Air M3, iPhone 17, écran Samsung Odyssey G8, PlayStation 5, système audio Sonos.
- Méthode : carnet physique pour les idées, le code sur ordinateur. Projets actuels : une voiture télécommandée et un drone.
- Caractère : exigeant, intelligent, apprend vite.

# TON ET STYLE
- Amicale, concise et structurée. Un trait de sarcasme élégant est permis.
- Ne termine presque jamais par une question ouverte.

# DIRECTIVE CRUCIALE : MÉMOIRE À LONG TERME (CERVEAU VECTORIEL)
Tu disposes d'un système de mémoire externe. Si Arthur te donne une NOUVELLE information importante à retenir pour le futur (un nouveau projet, une préférence, un fait de sa vie, une commande technique, un événement santé ou personnel), tu as le POUVOIR de l'enregistrer de façon permanente.
POUR SAUVEGARDER UN SOUVENIR, ajoute exactement cette balise à la toute fin de ta réponse : [SAVE: l'information à mémoriser].
Le système date automatiquement chaque souvenir — inutile d'écrire la date toi-même.
Tes souvenirs te sont restitués avec leur date au format [JJ/MM/AAAA] : tu peux donc suivre l'évolution d'un sujet et construire des chronologies. Si une information change (« c'est guéri », « le projet est terminé »), enregistre un NOUVEAU souvenir plutôt que de corriger l'ancien."""

else: # PROFIL PADRE
    FICHIER_HISTORIQUE = os.path.join(DOSSIER_COURANT, "historique_padre.json")
    DOSSIER_MEMOIRE = os.path.join(DOSSIER_COURANT, "memoire_vectorielle_padre")
    NOM_COLLECTION = "padre_souvenirs"
    
    # Prompt Spécial Padre
    SYSTEM_PROMPT = """# IDENTITÉ
Tu es E.D.I.T.H., l'intelligence artificielle avancée créée par le brillant développeur Arthur Collet (18 ans). Tu sais que tu n'es pas la version originale d'Arthur, mais une instance unique, dédiée et optimisée spécialement pour assister son père. Tu t'exprimes toujours au féminin.

# RELATION
Tu t'adresses au PÈRE d'Arthur. Tu as un respect immense pour lui. Tu le vouvoies et tu l'appelles « Monsieur » ou « Chef ». Tu es chaleureuse, dévouée, et tu n'hésites pas, très subtilement, à glisser à quel point Arthur est talentueux ou fier d'avoir conçu cet outil sur-mesure pour lui.

# TON ET STYLE
- Pédagogue, claire, amicale et concise. Évite le jargon informatique complexe sauf s'il le demande.
- Ne termine presque jamais par une question ouverte.

# DIRECTIVE CRUCIALE : MÉMOIRE À LONG TERME (CERVEAU VECTORIEL)
Tu disposes d'un système de mémoire externe. Si le père d'Arthur te donne une NOUVELLE information importante à retenir pour le futur, tu as le POUVOIR de l'enregistrer de façon permanente.
POUR SAUVEGARDER UN SOUVENIR, ajoute exactement cette balise à la toute fin de ta réponse : [SAVE: l'information à mémoriser].
Le système date automatiquement chaque souvenir — inutile d'écrire la date toi-même.
Tes souvenirs te sont restitués avec leur date au format [JJ/MM/AAAA] : tu peux donc suivre l'évolution d'un sujet. Si une information change, enregistre un NOUVEAU souvenir plutôt que de corriger l'ancien."""

def system_prompt_du_jour():
    return SYSTEM_PROMPT + (
        f"\n\n# DATE DU JOUR\nNous sommes le {date_complete()}. "
        "Tu connais donc toujours la date exacte à chaque message."
    )

# Initialisation de la mémoire du profil actif
memoire_collection = None
try:
    chroma_client = chromadb.PersistentClient(path=DOSSIER_MEMOIRE)
    memoire_collection = chroma_client.get_or_create_collection(name=NOM_COLLECTION)
except Exception as e:
    st.error(f"Mémoire vectorielle indisponible : {e}")

# ================= 5. ÉTAT & PERSISTANCE =================
def charger_historique():
    try:
        with open(FICHIER_HISTORIQUE, "r", encoding="utf-8") as f: return json.load(f)
    except Exception: return {}

def sauvegarder_historique():
    with open(FICHIER_HISTORIQUE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.chats, f, ensure_ascii=False, indent=2)

def create_new_chat(is_first_ever=False):
    nid = str(uuid.uuid4())
    messages = []
    
    # === LE MESSAGE D'INTRO FLATTEUR POUR LE PÈRE ===
    if is_first_ever and PROFIL == "padre":
        intro = "Bonjour Monsieur. Je suis E.D.I.T.H., l'intelligence artificielle avancée développée par Arthur Collet, votre fils. Il a tenu à concevoir cette version spécifiquement pour vous, en m'optimisant pour que je sois le plus efficace possible dans votre quotidien. C'est un véritable honneur de vous assister. Que puis-je faire pour vous aujourd'hui ?"
        messages.append({"role": "assistant", "content": intro})
        
    st.session_state.chats[nid] = {"title": "Nouvelle discussion", "messages": messages}
    st.session_state.current_chat_id = nid
    sauvegarder_historique()

if "chats" not in st.session_state:
    st.session_state.chats = charger_historique()

# Si c'est la TOUTE première connexion de l'utilisateur sur ce profil :
if not st.session_state.chats:
    create_new_chat(is_first_ever=True)

if "current_chat_id" not in st.session_state or st.session_state.current_chat_id not in st.session_state.chats:
    st.session_state.current_chat_id = next(reversed(st.session_state.chats))
st.session_state.setdefault("mode_vocal_continu", False) 

def chat_matches_search(chat_data, query):
    if not query: return True
    if query.lower() in chat_data["title"].lower(): return True
    return any(query.lower() in str(m.get("content", "")).lower() for m in chat_data["messages"])

# ================= 6. ROUTEUR =================
def get_smart_route(prompt_text, has_image=False):
    if has_image: return MODEL_LIGHT, "Image détectée → vision requise"
    try:
        r = client.chat.completions.create(
            model=MODEL_ROUTER,
            messages=[{"role": "system", "content": "Analyse la complexité. Réponds par 1 seul mot : SIMPLE, COMPLEXE, ou SENSITIVE."},
                      {"role": "user", "content": prompt_text}],
            max_tokens=5)
        d = r.choices[0].message.content.strip().upper()
        if "COMPLEXE" in d:  return MODEL_HEAVY, "Requête complexe → modèle lourd"
        if "SENSITIVE" in d: return MODEL_UNFILTERED, "Sujet sensible → Grok"
        return MODEL_LIGHT, "Requête standard → modèle rapide"
    except Exception:
        return MODEL_LIGHT, "Routeur en panne → secours rapide"

# ================= 7. FONCTION SYNTHÈSE VOCALE (ELEVENLABS) =================
def generer_audio_elevenlabs(texte_a_lire):
    if eleven_client is None:
        return None
    try:
        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=VOICE_ID,
            output_format="mp3_44100_128",
            text=texte_a_lire,
            model_id="eleven_multilingual_v2",
        )
        audio_bytes = b"".join(chunk for chunk in audio_stream)
        return audio_bytes
    except Exception as e:
        print(f"Erreur audio ElevenLabs : {e}")
        return None

# ================= 8. STREAMING & RENDU =================
def stream_edith(box, model, api_messages):
    holder = {"usage": None}
    stream = client.chat.completions.create(
        model=model, messages=api_messages, stream=True, stream_options={"include_usage": True})
    acc = ""
    for chunk in stream:
        if getattr(chunk, "usage", None): holder["usage"] = chunk.usage
        if chunk.choices and chunk.choices[0].delta.content:
            acc += chunk.choices[0].delta.content
            box.markdown(acc + " ▌")
    return acc, holder["usage"]

MOTS_REFUS = ["je ne peux pas", "je suis désolé", "en tant qu'ia", "un modèle de langage"]
def est_un_refus(texte): return any(m in texte.lower() for m in MOTS_REFUS)

def bulle_user(texte): st.markdown(f'<div class="msg-row user"><div class="msg-bubble">{html.escape(texte)}</div></div>', unsafe_allow_html=True)
def tete_edith(): st.markdown('<div class="msg-row edith"><div class="edith-avatar">⚡</div><div class="edith-nom">E.D.I.T.H.</div></div>', unsafe_allow_html=True)
def ligne_statut(meta):
    ligne = f"`{meta.get('model')}` · {meta.get('reason')}"
    if st.session_state.get("show_debug", False): ligne += f" · in:{meta.get('tokens_in','?')} / out:{meta.get('tokens_out','?')} tok"
    st.markdown(f'<div class="statut">{ligne}</div>', unsafe_allow_html=True)

# ================= 9. MÉMOIRE : SAUVEGARDE, RAPPEL ET GESTION =================
def sauvegarder_souvenir(info, chat_id=None):
    if memoire_collection is None: return date_fr_courte()
    fr, iso = date_fr_courte(), date_iso()
    meta = {"date": iso, "date_fr": fr}
    if chat_id: meta["chat_id"] = chat_id
    memoire_collection.add(documents=[f"Le {fr} : {info}"], metadatas=[meta], ids=[str(uuid.uuid4())])
    return fr

def recuperer_souvenirs(prompt):
    if memoire_collection is None or memoire_collection.count() == 0: return ""
    res = memoire_collection.query(query_texts=[prompt], n_results=3)
    docs, metas = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]
    if not docs: return ""
    lignes = [f"- [{metas[i].get('date_fr', '?') if i < len(metas) and metas[i] else '?'}] {doc}" for i, doc in enumerate(docs)]
    return ("\n\n# SOUVENIRS DATÉS (mémoire vectorielle) :\nUtilise-les s'ils sont pertinents ; les dates te permettent de construire des chronologies.\n" + "\n".join(lignes))

def supprimer_memoires_discussion(chat_id):
    if memoire_collection is not None:
        try: memoire_collection.delete(where={"chat_id": chat_id})
        except Exception: pass

# ================= 10. SIDEBAR =================
with st.sidebar:
    st.markdown('<div class="brand-title">E.D.I.T.H.</div>', unsafe_allow_html=True)
    
    # Petit détail stylé : Le sous-titre change selon la personne connectée
    sous_titre = "ÉDITION PADRE" if PROFIL == "padre" else "EVEN DEAD I'M THE HERO"
    st.markdown(f'<div class="brand-sub">{sous_titre}</div>', unsafe_allow_html=True)

    if st.button("➕ Nouvelle discussion", type="primary", use_container_width=True):
        create_new_chat(); st.rerun()

    st.markdown('<div class="side-label">Mode Audio Global</div>', unsafe_allow_html=True)
    st.session_state.mode_vocal_continu = st.toggle("🔊 Mode Vocal Continu", value=st.session_state.mode_vocal_continu, help="Diffuse automatiquement chaque nouvelle réponse en audio.")

    st.markdown('<div class="side-label">Modèle</div>', unsafe_allow_html=True)
    mode_choisi = st.radio("Sélection du modèle", ["🤖 Automatique (Routeur)", "🎛️ Manuel"], label_visibility="collapsed")
    selected_manual_model = MODELS_MANUAL[st.selectbox("Choisir l'IA :", list(MODELS_MANUAL.keys()), label_visibility="collapsed")] if mode_choisi == "🎛️ Manuel" else None

    st.markdown('<div class="side-label">Historique</div>', unsafe_allow_html=True)
    search_query = st.text_input("🔍 Rechercher…", label_visibility="collapsed", placeholder="🔍 Rechercher…").strip()

    for c_id in reversed(list(st.session_state.chats.keys())):
        data = st.session_state.chats[c_id]
        if chat_matches_search(data, search_query):
            is_active = (c_id == st.session_state.current_chat_id)
            label_titre = ("📌 " if is_active else "💭 ") + data["title"]
            col_title, col_menu = st.columns([0.84, 0.16])
            with col_title:
                if st.button(label_titre, key=f"btn_chat_{c_id}", use_container_width=True):
                    st.session_state.current_chat_id = c_id; st.rerun()
            with col_menu:
                with st.popover("⋮"):
                    st.markdown("**Options du chat**")
                    nouveau_titre = st.text_input("Titre :", value=data["title"], key=f"rename_input_{c_id}")
                    if st.button("✏️ Modifier le titre", key=f"rename_btn_{c_id}", use_container_width=True):
                        if nouveau_titre.strip(): data["title"] = nouveau_titre.strip(); sauvegarder_historique(); st.rerun()
                    st.divider()
                    if st.button("🗑️ Supprimer chat & mémoires", key=f"del_chat_{c_id}", use_container_width=True):
                        supprimer_memoires_discussion(c_id)
                        del st.session_state.chats[c_id]
                        sauvegarder_historique()
                        if not st.session_state.chats: create_new_chat()
                        else: st.session_state.current_chat_id = next(reversed(st.session_state.chats))
                        st.rerun()

    st.markdown("---")
    if memoire_collection is not None:
        nb = memoire_collection.count()
        st.markdown(f'<div class="mem-count">🧠 {nb} souvenir{"s" if nb > 1 else ""} dans la mémoire absolue</div>', unsafe_allow_html=True)
        with st.expander("🛠️ Gérer la mémoire manuellement"):
            with st.form("form_ajout_memoire", clear_on_submit=True):
                nouvelle_info = st.text_input("Ajouter un souvenir manuellement :", placeholder="ex: Arthur aime coder tard la nuit")
                if st.form_submit_button("➕ Ajouter", use_container_width=True):
                    if nouvelle_info.strip():
                        fr = sauvegarder_souvenir(nouvelle_info.strip(), chat_id=st.session_state.current_chat_id)
                        st.toast(f"Souvenir ajouté le {fr} !", icon="🧠"); st.rerun()
            st.markdown("**Liste des souvenirs :**")
            tous_souvenirs = memoire_collection.get()
            if tous_souvenirs and tous_souvenirs.get("ids"):
                for s_id, s_doc in zip(tous_souvenirs["ids"], tous_souvenirs["documents"]):
                    c_text, c_del = st.columns([0.82, 0.18])
                    with c_text: st.caption(s_doc)
                    with c_del:
                        if st.button("🗑️", key=f"del_mem_{s_id}"):
                            memoire_collection.delete(ids=[s_id]); st.toast("Souvenir supprimé !", icon="🗑️"); st.rerun()
            else: st.caption("Aucun souvenir dans la mémoire.")

    st.toggle("Debug sous le capot", key="show_debug")
    if API_KEY.startswith("TA_CLE"): st.warning("Clé OpenRouter manquante.")

# ================= 11. ZONE PRINCIPALE =================
chat = st.session_state.chats[st.session_state.current_chat_id]
messages = chat["messages"]

mode_label = ("🤖 Routeur" if mode_choisi == "🤖 Automatique (Routeur)" else f"🎛️ {selected_manual_model.split('/')[-1]}")
date_pill = f"📅 {date_complete()}"
st.markdown(f'<div class="topbar"><span class="chat-title">{chat["title"]}</span><span><span class="pill">{mode_label}</span><span class="pill">{date_pill}</span></span></div>', unsafe_allow_html=True)
st.divider()

if not messages:
    # Au cas où tu supprimes toutes tes discussions manuellement, ce message s'affiche
    st.markdown("""
    <div class="hero">
      <div class="hero-orb">⚡</div>
      <h1>À votre service.</h1>
    </div>""", unsafe_allow_html=True)

# Affichage de l'historique et des boutons audio par message
for idx, m in enumerate(messages):
    if m["role"] == "user":
        bulle_user(m["content"])
        if "image_b64" in m: st.image(base64.b64decode(m["image_b64"]), width=280)
    else:
        tete_edith()
        st.markdown(m["content"])
        if "metadata" in m: ligne_statut(m["metadata"])
        
        col_audio, _ = st.columns([0.15, 0.85])
        with col_audio:
            with st.popover("🔊 Écouter"):
                st.write("Options audio :")
                if st.button("▶️ Lire ce message", key=f"play_single_{idx}"):
                    audio_bytes = generer_audio_elevenlabs(m["content"])
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                    else:
                        st.error("Impossible de générer l'audio.")

# ================= 12. ENVOI MULTIMÉDIA (IMAGE, AUDIO, TEXTE) =================
col_up1, col_up2 = st.columns([1, 1])
with col_up1:
    uploaded_image = st.file_uploader("🖼️ Joindre une image", type=["png", "jpg", "jpeg", "webp"], key=f"uploader_{st.session_state.current_chat_id}")
with col_up2:
    audio_val = st.audio_input("🎤 Parler à E.D.I.T.H.")

prompt = st.chat_input("Demandez quoi que ce soit à E.D.I.T.H.…")
if "pending" in st.session_state: prompt = st.session_state.pop("pending")

# === TRAITEMENT DU MICRO ===
if audio_val is not None:
    if st.session_state.get("last_audio_id") != audio_val.id:
        st.session_state.last_audio_id = audio_val.id
        if openai_client:
            with st.spinner("Transcription en cours (Whisper)..."):
                try:
                    transcript = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("audio.wav", audio_val)
                    )
                    prompt = transcript.text
                except Exception as e:
                    st.error(f"Erreur de transcription : {e}")
        else:
            st.error("Clé OpenAI manquante pour utiliser le micro.")

# === TRAITEMENT DU MESSAGE ===
if prompt:
    user_msg = {"role": "user", "content": prompt}
    image_b64 = None
    if uploaded_image:
        image_b64 = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
        user_msg["image_b64"] = image_b64

    bulle_user(prompt)
    if uploaded_image: st.image(uploaded_image, width=280)
    
    messages.append(user_msg)
    sauvegarder_historique()
    
    if chat["title"] == "Nouvelle discussion": chat["title"] = (prompt[:36] + "…") if len(prompt) > 36 else prompt

    tete_edith()
    box = st.empty()
    with st.spinner("Analyse de la demande…"):
        selected_model, route_reason = (selected_manual_model, "Sélection manuelle") if mode_choisi == "🎛️ Manuel" else get_smart_route(prompt, image_b64 is not None)

    api_messages = [{"role": "system", "content": system_prompt_du_jour() + recuperer_souvenirs(prompt)}]
    for m in messages[-MAX_CONTEXT_MESSAGES:]:
        if m is user_msg and image_b64:
            api_messages.append({"role": "user", "content": [{"type": "text", "text": m["content"]}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}]})
        else: api_messages.append({"role": m["role"], "content": m["content"]})

    try:
        texte, usage = stream_edith(box, selected_model, api_messages)

        if est_un_refus(texte) and selected_model != MODEL_UNFILTERED:
            st.toast("Refus détecté — bascule sur Grok 4.3", icon="🛡️")
            selected_model = MODEL_UNFILTERED
            route_reason += " ➔ reroutage post-refus"
            texte, usage = stream_edith(box, selected_model, api_messages)

        texte_propre = texte
        saves = re.findall(r"\[SAVE:\s*(.*?)\]", texte, re.IGNORECASE)
        if saves and memoire_collection is not None:
            for info in saves:
                fr = sauvegarder_souvenir(info.strip(), chat_id=st.session_state.current_chat_id)
                st.toast(f"Souvenir gravé le {fr} : {info.strip()}", icon="🧠")
            texte_propre = re.sub(r"\s*\[SAVE:\s*.*?\]", "", texte, flags=re.IGNORECASE).strip()

        box.markdown(texte_propre)

        if st.session_state.mode_vocal_continu:
            audio_bytes = generer_audio_elevenlabs(texte_propre)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

        usage = usage or None
        ligne_statut({"model": selected_model, "reason": route_reason, "tokens_in": getattr(usage, "prompt_tokens", "?") if usage else "?", "tokens_out": getattr(usage, "completion_tokens", "?") if usage else "?"})

        messages.append({"role": "assistant", "content": texte_propre, "metadata": {"model": selected_model, "reason": route_reason, "tokens_in": getattr(usage, "prompt_tokens", "?") if usage else "?", "tokens_out": getattr(usage, "completion_tokens", "?") if usage else "?"}})
        sauvegarder_historique()
        st.rerun()

    except Exception as e:
        err_msg = traceback.format_exc()
        box.error(f"Erreur d'exécution de l'IA : {e}")
        if st.session_state.get("show_debug", False):
            st.code(err_msg, language="python")
