import importlib
import io
import sys

import pytest


@pytest.mark.parametrize("module_name", ["main", "mcp_agent"])
def test_cli_can_print_unicode_model_responses_on_windows_cp1252(monkeypatch, module_name):
    module = importlib.import_module(module_name)
    reponse = "Réponse documentaire 🟦 [S1]"
    if module_name == "main":
        monkeypatch.setattr(module.RAGPipeline, "rag", lambda *args, **kwargs: {
            "reponse": reponse, "sources_transmises": [], "avertissements": [],
        })
        arguments = ["main.py", "ask", "Question"]
    else:
        async def interroger(*args):
            return {"texte": reponse, "sources": [], "appels_outils": [], "avertissements": []}
        monkeypatch.setattr(module, "interroger_mcp", interroger)
        arguments = ["mcp_agent.py", "Question"]
    with io.TextIOWrapper(io.BytesIO(), encoding="cp1252") as sortie:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stdout", sortie)
            patch.setattr(sys, "argv", arguments)
            module.main()
            sortie.flush()
            contenu = sortie.buffer.getvalue()
    assert reponse in contenu.decode("utf-8")
