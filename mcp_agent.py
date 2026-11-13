"""Client agent séparé : aucune importation de CLIP, Chroma ou RAGPipeline."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Literal

from langchain.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from rag_agent import analyser_execution, creer_agent, message_utilisateur


async def interroger_mcp(texte=None, image=None, type_resultat="tous", k=6, racine=None):
    """Garder une session MCP ouverte pendant toute la boucle d'outils de l'agent."""
    message = message_utilisateur(texte, image, type_resultat, k)
    environnement = dict(os.environ)
    if racine is not None:
        environnement["RAG_DATA_DIR"] = str(Path(racine).resolve())
    serveur = StdioServerParameters(
        command=sys.executable, args=[str(Path(__file__).with_name("mcp_server.py"))],
        env=environnement,
    )
    async with stdio_client(serveur) as (lecture, ecriture):
        async with ClientSession(lecture, ecriture) as session:
            await session.initialize()
            outils = await session.list_tools()
            if "consulter_pdf" not in {outil.name for outil in outils.tools}:
                raise RuntimeError("Le serveur MCP ne fournit pas consulter_pdf.")

            @tool
            async def consulter_pdf(question: str, type_resultat: Literal["tous", "texte", "image"] = "tous", k: int = 6) -> str:
                """Consulter les PDF via MCP pour retrouver explications, chiffres et figures."""
                resultat = await session.call_tool("consulter_pdf", {
                    "question": question, "type_resultat": type_resultat, "k": k,
                })
                if resultat.isError:
                    texte_erreur = "\n".join(bloc.text for bloc in resultat.content if bloc.type == "text")
                    raise RuntimeError(f"Erreur du RAG MCP : {texte_erreur}")
                if resultat.structuredContent is not None:
                    return json.dumps(resultat.structuredContent, ensure_ascii=False)
                return "\n".join(bloc.text for bloc in resultat.content if bloc.type == "text")

            agent = creer_agent([consulter_pdf])
            execution = await agent.ainvoke({"messages": [message]}, config={"recursion_limit": 20})
    # Le client ne lit pas les chemins ni les images du serveur. Les citations
    # et les identifiants documentaires suffisent pour la réponse distante.
    return analyser_execution(execution, type_resultat=type_resultat)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Agent documentaire utilisant le RAG via MCP stdio")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--type", choices=["tous", "texte", "image"], default="tous")
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    sortie = asyncio.run(interroger_mcp(args.question, args.image, args.type, args.k, args.data_dir))
    print(sortie["texte"])
    print(json.dumps({"sources": sortie["sources"], "appels_outils": sortie["appels_outils"],
                      "avertissements": sortie["avertissements"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
