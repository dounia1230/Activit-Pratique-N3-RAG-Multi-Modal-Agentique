"""Interface en ligne de commande pour indexer et interroger les PDF."""

import argparse
import json
from pathlib import Path

from rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="RAG multimodal local")
    parser.add_argument("--data-dir", type=Path, default=None)
    commands = parser.add_subparsers(dest="commande", required=True)
    indexer = commands.add_parser("index", help="Extraire et indexer un PDF")
    indexer.add_argument("pdf", type=Path)
    for nom in ("ask", "agent"):
        commande = commands.add_parser(nom, help="Interroger directement le RAG" if nom == "ask" else "Interroger l'agent LangChain")
        commande.add_argument("question", nargs="?")
        commande.add_argument("--image", type=Path)
        commande.add_argument("--type", choices=["tous", "texte", "image"], default="tous")
        commande.add_argument("-k", type=int, default=6)
    args = parser.parse_args()
    pipeline = RAGPipeline(args.data_dir)
    if args.commande == "index":
        print(json.dumps(pipeline.indexer_pdf(args.pdf), ensure_ascii=False, indent=2))
    elif args.commande == "ask":
        question = args.question or ("Explique cette image à partir des sources documentaires." if args.image else "")
        sortie = pipeline.rag(question, k=args.k, image_requete=args.image,
                              type_resultat=None if args.type == "tous" else args.type)
        print(sortie["reponse"])
        for source in sortie["sources_transmises"]:
            print(f"[{source['reference']}] {source['pdf']}, p. {source['page']} ({source['type']})")
        for avertissement in sortie["avertissements"]:
            print("Avertissement :", avertissement)
    else:
        from rag_agent import creer_agent, creer_outil, interroger_agent
        agent = creer_agent([creer_outil(pipeline.rag)])
        sortie = interroger_agent(agent, pipeline.collection, args.question, args.image, args.type, args.k)
        print(sortie["texte"])
        print(json.dumps(sortie["appels_outils"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
