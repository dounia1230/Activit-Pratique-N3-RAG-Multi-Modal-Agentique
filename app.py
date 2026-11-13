"""Interface Streamlit locale : ingestion, RAG et agent multimodal."""

import hashlib
from pathlib import Path

import streamlit as st

from rag_agent import creer_agent, creer_outil, interroger_agent
from rag_pipeline import RAGPipeline


@st.cache_resource
def charger_pipeline():
    return RAGPipeline()


def sauvegarder_upload(upload, dossier):
    donnees = upload.getvalue()
    # Le nom utilisateur ne peut pas sortir du dossier des uploads.
    chemin = dossier / hashlib.sha256(donnees).hexdigest()[:24] / Path(upload.name).name
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_bytes(donnees)
    return chemin


def main():
    st.set_page_config(page_title="RAG multimodal documentaire", layout="wide")
    st.title("Interroger vos PDF")
    st.write("Posez une question ou fournissez une image et consultez les sources de la réponse.")
    pipeline = charger_pipeline()
    with st.sidebar:
        st.header("Documents")
        pdf = st.file_uploader("Ajouter un PDF", type=["pdf"])
        if st.button("Indexer le document", disabled=pdf is None):
            try:
                with st.spinner("Extraction et indexation du PDF…"):
                    chemin = sauvegarder_upload(pdf, pipeline.racine / "uploads")
                    bilan = pipeline.indexer_pdf(chemin)
                st.success(f"{bilan['textes']} passages et {bilan['images']} images indexés.")
                for message in bilan["avertissements"]:
                    st.warning(message)
            except Exception as erreur:
                st.error(f"Indexation impossible : {erreur}")
        mode = st.radio("Mode", ["RAG direct", "Agent LangChain"])
        filtre = st.selectbox("Sources à rechercher", ["tous", "texte", "image"])
        k = st.slider("Nombre de sources", 1, 12, 6)
        st.caption("Démarrer Ollama et installer le modèle avant la génération.")

    with st.form("question"):
        question = st.text_area("Votre question", placeholder="Pourquoi utiliser plusieurs têtes d’attention ?")
        image = st.file_uploader("Image de recherche (facultative)", type=["png", "jpg", "jpeg", "webp"])
        envoyer = st.form_submit_button("Rechercher et répondre")
    if envoyer:
        if not question.strip() and image is None:
            st.info("Saisissez une question ou ajoutez une image.")
        else:
            try:
                image_path = sauvegarder_upload(image, pipeline.racine / "uploads") if image else None
                with st.spinner("Recherche dans les documents et préparation de la réponse…"):
                    if mode == "RAG direct":
                        sortie = pipeline.rag(question.strip() or "Explique cette image avec les sources du PDF.",
                                              image_requete=image_path, type_resultat=None if filtre == "tous" else filtre, k=k)
                        resultat = {"texte": sortie["reponse"], "sources": sortie["sources_transmises"],
                                    "avertissements": sortie["avertissements"], "appels_outils": [], "images": []}
                    else:
                        agent = creer_agent([creer_outil(pipeline.rag)])
                        resultat = interroger_agent(agent, pipeline.collection, question, image_path, filtre, k)
                st.session_state["resultat"] = resultat
            except Exception as erreur:
                st.session_state.pop("resultat", None)
                st.error(f"La demande a échoué : {erreur}")

    if "resultat" in st.session_state:
        resultat = st.session_state["resultat"]
        st.markdown(resultat["texte"])
        for message in resultat["avertissements"]:
            st.warning(message)
        if resultat["sources"]:
            st.subheader("Sources")
            st.dataframe([{cle: source.get(cle) for cle in ("reference", "pdf", "page", "type", "appel_id")}
                          for source in resultat["sources"]], hide_index=True)
        for source in resultat["sources"]:
            if source.get("type") == "image" and source.get("source"):
                chemin = Path(source["source"])
                if chemin.is_file():
                    st.image(str(chemin), caption=f"{source['pdf']}, p. {source['page']}")
        for image in resultat["images"]:
            st.image(image["image"], caption=f"{image['pdf']}, p. {image['page']}")
        if resultat["appels_outils"]:
            with st.expander("Recherches effectuées par l’agent"):
                st.json(resultat["appels_outils"])


if __name__ == "__main__":
    main()
