"""Sign vocabulary shared by the simulator, the text-importance training data and the taxonomy evaluation.

`research/sign_taxonomy.json` lists 89 illustrative sign strings in 13 categories (English). Blind users travel
through airports and stations where the same intent is printed in other languages, so a small translation table
covers the core wayfinding and hazard intents in Portuguese, Spanish and French. These are open-vocabulary
OCR targets, not verified communication labels.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# intent -> language -> sign text (English first)
SIGN_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "exit":            {"en": "EXIT", "pt": "SAÍDA", "es": "SALIDA", "fr": "SORTIE"},
    "entrance":        {"en": "ENTRANCE", "pt": "ENTRADA", "es": "ENTRADA", "fr": "ENTRÉE"},
    "emergency exit":  {"en": "EMERGENCY EXIT", "pt": "SAÍDA DE EMERGÊNCIA", "es": "SALIDA DE EMERGENCIA", "fr": "SORTIE DE SECOURS"},
    "stairs":          {"en": "STAIRS", "pt": "ESCADAS", "es": "ESCALERAS", "fr": "ESCALIER"},
    "elevator":        {"en": "ELEVATOR", "pt": "ELEVADOR", "es": "ASCENSOR", "fr": "ASCENSEUR"},
    "escalator":       {"en": "ESCALATOR", "pt": "ESCADA ROLANTE", "es": "ESCALERA MECÁNICA", "fr": "ESCALIER MÉCANIQUE"},
    "restroom":        {"en": "RESTROOM", "pt": "BANHEIRO", "es": "BAÑOS", "fr": "TOILETTES"},
    "do not enter":    {"en": "DO NOT ENTER", "pt": "NÃO ENTRE", "es": "NO ENTRAR", "fr": "ENTRÉE INTERDITE"},
    "wet floor":       {"en": "CAUTION WET FLOOR", "pt": "CUIDADO PISO MOLHADO", "es": "CUIDADO PISO MOJADO", "fr": "ATTENTION SOL GLISSANT"},
    "watch your step": {"en": "WATCH YOUR STEP", "pt": "CUIDADO COM O DEGRAU", "es": "CUIDADO CON EL ESCALÓN", "fr": "ATTENTION À LA MARCHE"},
    "gates":           {"en": "GATES", "pt": "PORTÕES", "es": "PUERTAS DE EMBARQUE", "fr": "PORTES D'EMBARQUEMENT"},
    "gate":            {"en": "GATE", "pt": "PORTÃO", "es": "PUERTA", "fr": "PORTE"},
    "departures":      {"en": "DEPARTURES", "pt": "PARTIDAS", "es": "SALIDAS", "fr": "DÉPARTS"},
    "arrivals":        {"en": "ARRIVALS", "pt": "CHEGADAS", "es": "LLEGADAS", "fr": "ARRIVÉES"},
    "baggage claim":   {"en": "BAGGAGE CLAIM", "pt": "RESTITUIÇÃO DE BAGAGEM", "es": "RECOGIDA DE EQUIPAJE", "fr": "LIVRAISON DES BAGAGES"},
    "security check":  {"en": "SECURITY CHECK", "pt": "CONTROLE DE SEGURANÇA", "es": "CONTROL DE SEGURIDAD", "fr": "CONTRÔLE DE SÛRETÉ"},
    "passport control": {"en": "PASSPORT CONTROL", "pt": "CONTROLE DE PASSAPORTES", "es": "CONTROL DE PASAPORTES", "fr": "CONTRÔLE DES PASSEPORTS"},
    "platform":        {"en": "PLATFORM", "pt": "PLATAFORMA", "es": "ANDÉN", "fr": "QUAI"},
    "bus stop":        {"en": "BUS STOP", "pt": "PONTO DE ÔNIBUS", "es": "PARADA DE AUTOBÚS", "fr": "ARRÊT DE BUS"},
    "ticket office":   {"en": "TICKET OFFICE", "pt": "BILHETERIA", "es": "TAQUILLA", "fr": "BILLETTERIE"},
    "information":     {"en": "INFORMATION", "pt": "INFORMAÇÕES", "es": "INFORMACIÓN", "fr": "INFORMATIONS"},
    "help desk":       {"en": "HELP DESK", "pt": "BALCÃO DE ATENDIMENTO", "es": "ATENCIÓN AL CLIENTE", "fr": "ACCUEIL"},
    "pharmacy":        {"en": "PHARMACY", "pt": "FARMÁCIA", "es": "FARMACIA", "fr": "PHARMACIE"},
    "emergency":       {"en": "EMERGENCY", "pt": "EMERGÊNCIA", "es": "URGENCIAS", "fr": "URGENCES"},
    "checkout":        {"en": "CHECKOUT", "pt": "CAIXA", "es": "CAJA", "fr": "CAISSE"},
    "closed":          {"en": "CLOSED", "pt": "FECHADO", "es": "CERRADO", "fr": "FERMÉ"},
    "detour":          {"en": "DETOUR", "pt": "DESVIO", "es": "DESVÍO", "fr": "DÉVIATION"},
    "mind the gap":    {"en": "MIND THE GAP", "pt": "CUIDADO COM O VÃO", "es": "CUIDADO CON EL HUECO", "fr": "ATTENTION À L'ÉCART"},
}
# decorative / advertising text (should score low) in the same languages
DECOR_TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "sale":        {"en": "SALE 50% OFF", "pt": "PROMOÇÃO 50% OFF", "es": "REBAJAS 50%", "fr": "SOLDES 50%"},
    "welcome":     {"en": "WELCOME", "pt": "BEM-VINDO", "es": "BIENVENIDO", "fr": "BIENVENUE"},
    "menu":        {"en": "NEW MENU", "pt": "NOVO CARDÁPIO", "es": "NUEVO MENÚ", "fr": "NOUVEAU MENU"},
    "duty free":   {"en": "DUTY FREE SHOPPING", "pt": "COMPRAS DUTY FREE", "es": "COMPRAS LIBRES DE IMPUESTOS", "fr": "BOUTIQUES HORS TAXES"},
    "free wifi":   {"en": "FREE WIFI", "pt": "WIFI GRÁTIS", "es": "WIFI GRATIS", "fr": "WIFI GRATUIT"},
    "coffee":      {"en": "COFFEE $2", "pt": "CAFÉ R$5", "es": "CAFÉ 2€", "fr": "CAFÉ 2€"},
    "follow us":   {"en": "FOLLOW US ONLINE", "pt": "SIGA-NOS", "es": "SÍGUENOS", "fr": "SUIVEZ-NOUS"},
    "happy hour":  {"en": "HAPPY HOUR 5 TO 7", "pt": "HAPPY HOUR 17H ÀS 19H", "es": "HORA FELIZ 5 A 7", "fr": "HAPPY HOUR 17H À 19H"},
}
LANGUAGES = ("en", "pt", "es", "fr")


def translate(intent: str, lang: str = "en") -> str:
    table = SIGN_TRANSLATIONS.get(intent) or DECOR_TRANSLATIONS.get(intent)
    if table is None:
        return intent.upper()
    return table.get(lang, table["en"])


def load_taxonomy(path=None) -> Dict[str, List[str]]:
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "research" / "sign_taxonomy.json"
    return json.loads(p.read_text())["categories"]


# taxonomy categories whose strings are action-relevant for a traveller (1) versus not (0); symbols have no text
TAXONOMY_LABEL = {"directional": 1, "vertical_access": 1, "floor_hazard": 1, "restriction": 1, "emergency": 1, "airport": 1,
                  "transit": 1, "personal_services": 1, "street": 1, "medical": 1, "retail": 1, "negative_text": 0}
RETAIL_DECOR = {"SALE 50% OFF"}


def taxonomy_rows(path=None) -> List[Dict]:
    """(text, label, category, language) rows: the English taxonomy plus the translation table."""
    rows = []
    for cat, strings in load_taxonomy(path).items():
        if cat not in TAXONOMY_LABEL:
            continue
        for s in strings:
            rows.append({"text": s, "label": 0 if s in RETAIL_DECOR else TAXONOMY_LABEL[cat], "category": cat, "language": "en"})
    for intent, table in SIGN_TRANSLATIONS.items():
        for lang, s in table.items():
            if lang != "en":
                rows.append({"text": s, "label": 1, "category": f"translated:{intent}", "language": lang})
    for intent, table in DECOR_TRANSLATIONS.items():
        for lang, s in table.items():
            rows.append({"text": s, "label": 0, "category": f"decor:{intent}", "language": lang})
    return rows
