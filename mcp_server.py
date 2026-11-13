"""Exposer le RAG par MCP stdio, sans charger CLIP pendant la découverte."""

import logging
from functools import lru_cache
from typing import Literal

from mcp.server.fastmcp import FastMCP

from rag_pipeline import RAGPipeline, entier_borne, sortie_publique

mcp = FastMCP("RAG multimodal documentaire")


@lru_cache(maxsize=1)
def get_pipeline():
    return RAGPipeline()


@mcp.tool()
def consulter_pdf(question: str, type_resultat: Literal["tous", "texte", "image"] = "tous", k: int = 6) -> dict:
    """Rechercher dans les PDF indexés et répondre avec sources, figures et avertissements."""
    entier_borne(k, "k", 1, 12)
    if type_resultat not in ("tous", "texte", "image"):
        raise ValueError("Filtre de recherche invalide.")
    sortie = get_pipeline().rag(question=question, k=k,
                                type_resultat=None if type_resultat == "tous" else type_resultat)
    return sortie_publique(sortie)


if __name__ == "__main__":
    # Initialiser les bibliothèques natives avant les threads du transport Windows.
    # Cela ouvre uniquement les modules ; ni index ni poids CLIP ne sont chargés.
    import chromadb  # noqa: F401
    # stdout appartient au protocole MCP ; les journaux passent par stderr.
    logging.basicConfig(level=logging.WARNING)
    mcp.run(transport="stdio")
