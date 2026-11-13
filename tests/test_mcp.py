import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import mcp_server


def test_mcp_tool_uses_same_rag_contract(monkeypatch):
    class Pipeline:
        def rag(self, **kwargs):
            assert kwargs == {"question": "Question", "k": 3, "type_resultat": "texte"}
            return {"question": "Question", "reponse": "Réponse [S1]",
                    "sources_transmises": [{"reference": "S1", "id": "a", "pdf": "a.pdf",
                                            "page": 1, "type": "texte", "source": "private.txt"}],
                    "avertissements": []}
    monkeypatch.setattr(mcp_server, "get_pipeline", lambda: Pipeline())
    sortie = mcp_server.consulter_pdf("Question", "texte", 3)
    assert sortie["reponse_rag"] == "Réponse [S1]"
    assert "private" not in json.dumps(sortie)


def test_real_stdio_discovery_and_empty_index_error(tmp_path):
    async def scenario():
        params = StdioServerParameters(command=sys.executable,
            args=[str(Path(__file__).resolve().parents[1] / "mcp_server.py")],
            env={**os.environ, "RAG_DATA_DIR": str(tmp_path)})
        async with stdio_client(params) as (lecture, ecriture):
            async with ClientSession(lecture, ecriture) as session:
                await session.initialize()
                outils = await session.list_tools()
                outil = next(o for o in outils.tools if o.name == "consulter_pdf")
                assert "question" in outil.inputSchema["properties"]
                result = await session.call_tool("consulter_pdf", {"question": "Question"})
                assert result.isError
                assert "vide" in "".join(c.text for c in result.content if c.type == "text")
    asyncio.run(asyncio.wait_for(scenario(), timeout=45))
